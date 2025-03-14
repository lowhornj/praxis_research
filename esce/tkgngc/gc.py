import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch.distributions.normal import Normal
from tkgngc.utils import vae_loss, gumbel_softmax
import pandas as pd
from utils.utils import set_seed
import numpy as np

set_seed()

def weights(length):
    linArr = np.arange(1, length+1)
    arrSum = length * (length+1) // 2
    return np.flip(linArr/arrSum)

def linear_annealing(epoch, start_beta, end_beta, total_epochs):

    beta = start_beta + (end_beta - start_beta) * epoch / total_epochs

    return beta

#https://github.com/haofuml/cyclical_annealing/tree/master
def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L 

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

def create_lagged_data(time_series_data, num_lags=5):
    time_steps, num_entities = time_series_data.shape
    lagged_data = torch.empty((time_series_data.shape[0],num_lags-1,time_series_data.shape[1]))
    for lag in range(1,num_lags):
        if lag != 0:
            lagged = time_series_data[:-(lag),:]
            lagged = torch.unsqueeze(lagged, 1)
            lagged = F.pad(lagged, (0, 0, 0, 0, lag, 0))
            lagged_data[:,lag-1,:] = lagged[:,0,:]
    
    return lagged_data
    

class MultiHeadCausalAttention(nn.Module):
    """ Multi-head Attention for Causal Discovery """
    def __init__(self, dim_query, dim_key_value, dim_out, num_heads):
        super(MultiHeadCausalAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_head = dim_out // num_heads

        assert dim_out % num_heads == 0, "Output dimension must be divisible by the number of heads."

        self.query_proj = nn.Linear(dim_query, dim_out)
        self.key_proj = nn.Linear(dim_key_value, dim_out)
        self.value_proj = nn.Linear(dim_key_value, dim_out)

    def forward(self, query, key, value):
        Q = self.query_proj(query).view(query.size(0), -1, self.num_heads, self.dim_head).transpose(1, 2)
        Ks = []
        for k in range(key.shape[1]):
            n_key = key[:,k,:]
            Ks.append( self.key_proj(n_key).view(n_key.size(0), -1, self.num_heads, self.dim_head).transpose(1, 2))

        t_weights = weights(len(Ks))
        
        K = torch.zeros(Ks[0].size())
        for t, val, in enumerate(Ks):
             K = (val*t_weights[t])+K
        
        Vs = []
        for v in range(value.shape[1]):
            n_key = value[:,v,:]
            Vs.append( self.value_proj(n_key).view(n_key.size(0), -1, self.num_heads, self.dim_head).transpose(1, 2) )
        
        V = torch.zeros(Vs[0].size())
        for t, val, in enumerate(Vs):
             V = (val*t_weights[t])+V

        attn_weights = torch.matmul(Q, K.transpose(-1, -2)) / (self.dim_head ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, V).transpose(1, 2).reshape(query.size(0), -1, self.num_heads * self.dim_head)
        return attn_output, attn_weights

class AutoregressiveAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model) 
        self.scale = torch.sqrt(torch.tensor(d_model)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
    def forward(self, inputs):
        # Create query, key, and value matrices
        query = self.query_proj(inputs)
        key = self.key_proj(inputs)
        value = self.value_proj(inputs)
        
        # Calculate attention scores with causal masking 
        attention_scores = torch.matmul(query, key.transpose(1, 2)) / self.scale
        attention_mask = torch.triu(torch.ones(attention_scores.size()), diagonal=1).bool().to(attention_scores.device)
        attention_scores = attention_scores.masked_fill(attention_mask, -1e9)
        
        # Apply softmax and calculate weighted sum
        attention_weights = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, value)
        
        return output 

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
    """ Granger Causality Model with GAT and Multi-Head Attention """
    def __init__(self, pretrained_tkg, input_dim, hidden_dim, latent_dim, num_heads, num_nodes, num_lags, projection_dim=16):
        super(GrangerCausality, self).__init__()

        self.pretrained_tkg = pretrained_tkg
        self.num_nodes = num_nodes
        self.num_lags = num_lags

        # Graph Attention Encoder
        self.gat_encoder = GATv2Conv(input_dim + pretrained_tkg.entity_embedding.embedding_dim*2, hidden_dim, heads=num_heads)
        self.encoder_projection_layer = nn.Linear(hidden_dim * num_heads, projection_dim)
        
        # Multi-Head Attention in Encoder
        self.multihead_encoder = MultiHeadCausalAttention(projection_dim, input_dim, projection_dim, num_heads)
        
        # Variational Components
        self.encoder_fc = nn.Linear(projection_dim, latent_dim * 2)  # Mean and log variance

        # Multi-Head Attention in Decoder
        self.multihead_decoder = MultiHeadCausalAttention(latent_dim, input_dim, projection_dim, num_heads)
        self.decoder_projection_layer = nn.Linear(projection_dim, hidden_dim)
        
        # Graph Attention in Decoder
        self.gat_decoder = GATv2Conv(projection_dim, hidden_dim, heads=num_heads)
        
        # Fully Connected Layer to Reconstruct Input
        self.decoder_fc = nn.Linear(hidden_dim * num_heads, input_dim)

        A0=torch.ones(num_nodes,num_nodes)+torch.eye(num_nodes)
        A0 = A0/torch.sum(A0,1)

        # Hierarchical Adjacency Matrices
        self.direct_adjacency = nn.Parameter(A0)  # Direct causality
        self.indirect_adjacency = nn.Parameter(A0)  # Indirect causality

    def forward(self, entity_indices, relation_indices, timestamp_indices, time_series_data, edge_index, lagged_data):
        # Pretrained TKG embeddings
        entity_emb, relation_emb, _, timestamp_emb = self.pretrained_tkg(
            entity_indices, relation_indices, entity_indices, timestamp_indices
        )
        enriched_features = torch.cat([time_series_data, entity_emb, timestamp_emb], dim=-1)

        # Encoder: Graph Attention
        x = self.gat_encoder(enriched_features, edge_index)
        x = F.relu(x)

        # Projection Layer
        x = self.encoder_projection_layer(x)
        x = F.relu(x)

        # Encoder: Multi-Head Attention
        x, _ = self.multihead_encoder(x, lagged_data, lagged_data)
        x = F.relu(x)

        # Latent Space
        q_params = self.encoder_fc(x)
        mean, log_var = torch.chunk(q_params, 2, dim=-1)
        std = torch.exp(0.5 * log_var)
        z = mean + std * torch.randn_like(std)

        # Decoder: Multi-Head Attention
        x, _ = self.multihead_decoder(z, lagged_data, lagged_data)
        x = F.relu(x)

        #x = self.decoder_projection_layer(x)
        x = x.squeeze(1)

        # Decoder: Graph Attention
        x = self.gat_decoder(x, edge_index)
        x = F.relu(x)

        # Fully Connected Reconstruction
        x_reconstructed = self.decoder_fc(x)

        # Multi-Level Causal Effects
        direct_causal_effect = torch.einsum("ij,tj->ij", self.direct_adjacency, x_reconstructed)
        indirect_causal_effect = torch.einsum("ij,tlj->ij", self.indirect_adjacency, z)
        self.causal_effect = direct_causal_effect + indirect_causal_effect  # Weighted combination
        
        # Apply gumbel softmax for adjacency matrix sparsification
        self.adj = F.gumbel_softmax(self.causal_effect, tau=0.5, hard=True)
        #adj = F.sigmoid(causal_effect)

        return z, mean, log_var, x_reconstructed, self.causal_effect, self.adj

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
        sparsity_loss = sparsity_weight * torch.sum(torch.abs(self.causal_effect))

        # Total Loss
        return recon_loss + beta * kl_divergence + sparsity_loss


def train_model(data_class,pretrained_tkg,epochs=100):
    edge_index = data_class.edge_index
    entity_indices = data_class.entity_indices
    relation_indices = data_class.relation_indices
    timestamp_indices = data_class.timestamp_indices
    time_series_data = data_class.time_series_tensor
    lagged_data = create_lagged_data(time_series_data,5)
    # = lagged_data.to(torch.float32)
    model = GrangerCausality(pretrained_tkg ,input_dim=time_series_data.shape[1], hidden_dim=256, latent_dim=time_series_data.shape[1], num_heads=4, num_nodes=time_series_data.shape[1], num_lags=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    beta_schedule = frange_cycle_linear(epochs)
    for epoch in range(epochs):
        sparsity_weight = linear_annealing(epoch, start_beta=0.01, end_beta=1.0, total_epochs=epochs)
        beta = beta_schedule[epoch]
        optimizer.zero_grad()
        z, mean, log_var, x_reconstructed, causal_effect, adj = model(entity_indices, relation_indices, timestamp_indices, time_series_data, edge_index,lagged_data=lagged_data)
        
        loss = model.loss_function(x_reconstructed, time_series_data, mean, log_var, adj,beta=beta,sparsity_weight=sparsity_weight)
        loss.backward()
        optimizer.step()
    
    return z, mean, log_var, x_reconstructed, causal_effect, adj
