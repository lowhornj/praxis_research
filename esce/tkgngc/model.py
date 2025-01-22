import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch.distributions.normal import Normal
from tkgngc.utils import vae_loss
import pandas as pd

def create_lagged_data(time_series_tensor, num_lags):
    """
    Create lagged versions of a 2D time-series tensor.
    
    Args:
        time_series_tensor (torch.Tensor): Time-series data of shape [num_entities, time_steps].
        num_lags (int): Number of lagged steps to include.
        
    Returns:
        lagged_data (torch.Tensor): Lagged data of shape [num_entities, num_lags, time_steps - num_lags].
        current_data (torch.Tensor): Current data of shape [num_entities, time_steps - num_lags].
    """
    num_entities, time_steps = time_series_tensor.shape

    if time_steps <= num_lags:
        raise ValueError("Number of lags must be less than the number of time steps.")

    # Prepare lagged data
    lagged_data = []
    for lag in range(1, num_lags + 1):
        lagged_data.append(time_series_tensor[:, :-lag])  # Remove the last `lag` steps

    # Stack lagged data along a new dimension (num_lags)
    lagged_data = torch.stack(lagged_data, dim=1)  # Shape: [num_entities, num_lags, time_steps - num_lags]

    # Current data excludes the first `num_lags` time steps
    current_data = time_series_tensor[:, num_lags:]  # Shape: [num_entities, time_steps - num_lags]

    return lagged_data, current_data

class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        # get the shape of the tensor for the mean and log variance
        batch, dim = z_mean.shape
        # generate a normal random tensor (epsilon) with the same shape as z_mean
        # this tensor will be used for reparameterization trick
        epsilon = Normal(0, 1).sample((batch, dim)).to(z_mean.device)
        # apply the reparameterization trick to generate the samples in the
        # latent space
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

class CrossAttention(nn.Module):
    """Cross-Attention Module."""
    def __init__(self, dim_query, dim_key_value, dim_out, num_heads):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(dim_query, dim_out)
        self.key_proj = nn.Linear(dim_key_value, dim_out)
        self.value_proj = nn.Linear(dim_key_value, dim_out)
        self.num_heads = num_heads
        self.dim_head = dim_out // num_heads

        assert dim_out % num_heads == 0, "Output dimension must be divisible by the number of heads."

    def forward(self, query, key, value):
        # Linear projections
        Q = self.query_proj(query).view(query.size(0), -1, self.num_heads, self.dim_head).transpose(1, 2)
        K = self.key_proj(key).view(key.size(0), -1, self.num_heads, self.dim_head).transpose(1, 2)
        V = self.value_proj(value).view(value.size(0), -1, self.num_heads, self.dim_head).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-1, -2)) / (self.dim_head ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Attention output
        attn_output = torch.matmul(attn_weights, V).transpose(1, 2).reshape(query.size(0), -1, self.num_heads * self.dim_head)
        return attn_output, attn_weights

class GraphAttentionGC(nn.Module):
    """Graph Attention Network for Granger Causality."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        super(GraphAttentionGC, self).__init__()
        self.gat1 = GATv2Conv(input_dim, hidden_dim, heads=num_heads)
        self.gat2 = GATv2Conv(hidden_dim * num_heads, output_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        x = self.gat2(x, edge_index)
        return x

class GrangerCausality(nn.Module):
    """Granger Causality Model with GAT and Cross-Attention in the Decoder."""
    def __init__(self, pretrained_tkg, input_dim, hidden_dim, latent_dim, num_heads, num_nodes, num_lags,projection_dim = 16):
        super(GrangerCausality, self).__init__()

        self.pretrained_tkg = pretrained_tkg
        self.num_nodes = num_nodes
        self.num_lags = num_lags

        # Graph Attention Network in Encoder
        self.gat_encoder = GATv2Conv(input_dim + pretrained_tkg.entity_embedding.embedding_dim*2, hidden_dim, heads=num_heads)

        self.encoder_projection_layer = nn.Linear(hidden_dim * num_heads, projection_dim)
        
        # Cross-Attention in Encoder
        self.cross_attention_encoder = CrossAttention(
            dim_query=projection_dim,
            dim_key_value=projection_dim,
            dim_out=projection_dim,
            num_heads=num_heads,
        )

        # Variational Components
        self.encoder_fc = nn.Linear(projection_dim, latent_dim * 2)  # Mean and log variance

        # Cross-Attention in Decoder
        self.cross_attention_decoder = CrossAttention(
            dim_query=latent_dim,
            dim_key_value=projection_dim,
            dim_out=projection_dim,
            num_heads=num_heads,
        )

        self.decoder_projection_layer = nn.Linear(projection_dim,hidden_dim )

        # Graph Attention in Decoder
        self.gat_decoder = GATv2Conv(projection_dim, hidden_dim, heads=num_heads)

        # Fully Connected Layer to Reconstruct Input
        self.decoder_fc = nn.Linear(hidden_dim * num_heads, input_dim)

        # Adjacency Matrix for Causal Graph
        self.adjacency_matrix = nn.Parameter(torch.randn(num_nodes, num_nodes, num_lags))

    def forward(self, entity_indices, relation_indices, timestamp_indices, time_series_data, edge_index, lagged_data):
        """
        Forward pass for Granger causality detection with GAT and cross-attention in the decoder.
        Args:
            entity_indices (torch.Tensor): Indices of entities.
            relation_indices (torch.Tensor): Indices of relations.
            timestamp_indices (torch.Tensor): Indices of timestamps.
            time_series_data (torch.Tensor): Original time-series data.
            edge_index (torch.Tensor): Graph edges.
            lagged_data (torch.Tensor): Lagged time-series data of shape [batch, num_nodes, lagged_features].
        """
        # Pretrained TKG embeddings
        entity_emb, relation_emb, _, timestamp_emb = self.pretrained_tkg(
            entity_indices, relation_indices, entity_indices, timestamp_indices
        )
        #entity_emb = entity_emb.unsqueeze(-2).expand(-1, -1, time_series_data.size(-1))  # Match time steps

        # Concatenate TKG embeddings with time-series data
        enriched_features = torch.cat([time_series_data, entity_emb, timestamp_emb], dim=-1)  # [batch, num_nodes, input_dim + embedding_dim]

        # Encoder: Graph Attention
        x = self.gat_encoder(enriched_features, edge_index)
        x = F.relu(x)

        # Projection Layer
        x = self.encoder_projection_layer(x)
        x = F.relu(x)

        # Encoder: Cross-Attention
        x, j = self.cross_attention_encoder(x, lagged_data, lagged_data)
        x = F.relu(x)

        # Latent Space
        q_params = self.encoder_fc(x)
        mean, log_var = torch.chunk(q_params, 2, dim=-1)

        # Reparameterization Trick
        std = torch.exp(0.5 * log_var)
        z = mean + std * torch.randn_like(std)

        # Decoder: Cross-Attention
        x, _ = self.cross_attention_decoder(z, lagged_data, lagged_data)
        x = F.relu(x)

        # Projection Layer
        #x = self.decoder_projection_layer(x)
        #x = F.relu(x)
        x = x.squeeze(1)

        # Decoder: Graph Attention
        x = self.gat_decoder(x, edge_index)
        x = F.relu(x)

        # Decoder: Fully Connected Reconstruction
        x_reconstructed = self.decoder_fc(x)

        #if x.dim() == 2:
        #    x = x.unsqueeze(0)

        #adj = self.adjacency_matrix.mean(dim=2)

        causal_effect = torch.einsum("ijl,tj->ij",self.adjacency_matrix,x_reconstructed)

        # Learned Adjacency Matrix
        adj = torch.sigmoid(torch.abs(causal_effect))  # Values in [0, 1]

        return z, mean, log_var, x_reconstructed, causal_effect, adj

    def loss_function(self, x_reconstructed, time_series_data, mean, log_var, adj, sparsity_weight=0.01, beta=1.0):
        """
        Loss function for Granger causality with GAT and cross-attention in the decoder.
        Args:
            x_reconstructed (torch.Tensor): Reconstructed features.
            time_series_data (torch.Tensor): Original input features.
            mean (torch.Tensor): Mean of latent distribution.
            log_var (torch.Tensor): Log variance of latent distribution.
            adj (torch.Tensor): Learned adjacency matrix.
            sparsity_weight (float): Weight for sparsity regularization.
            beta (float): Weight for KL divergence.
        """
        # Reconstruction Loss
        recon_loss = F.mse_loss(x_reconstructed, time_series_data, reduction='sum')

        # KL Divergence
        kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        # Sparsity Regularization
        sparsity_loss = sparsity_weight * torch.sum(torch.abs(adj))

        # Total Loss
        return recon_loss + beta * kl_divergence + sparsity_loss


def train_model(data_class,pretrained_tkg):
    edge_index = data_class.edge_index
    entity_indices = data_class.entity_indices
    relation_indices = data_class.relation_indices
    timestamp_indices = data_class.timestamp_indices
    time_series_data = data_class.time_series_tensor
    lagged_data, original = create_lagged_data(time_series_data,1)

    model = GrangerCausality(pretrained_tkg ,input_dim=time_series_data.shape[1], hidden_dim=32, latent_dim=4, num_heads=4, num_nodes=time_series_data.shape[1], num_lags=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(50):
        optimizer.zero_grad()
        z, mean, log_var, x_reconstructed, causal_effect, adj = model(entity_indices, relation_indices, timestamp_indices, time_series_data, edge_index, lagged_data)
        loss = model.loss_function(x_reconstructed, time_series_data, mean, log_var, adj)
        loss.backward()
        optimizer.step()
    
    return z, mean, log_var, x_reconstructed, causal_effect, adj
