import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.distributions.normal import Normal

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
        K = self.key_proj(key).view(key.size(0), -1, self.num_heads, self.dim_head).transpose(1, 2)
        V = self.value_proj(value).view(value.size(0), -1, self.num_heads, self.dim_head).transpose(1, 2)

        attn_weights = torch.matmul(Q, K.transpose(-1, -2)) / (self.dim_head ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, V).transpose(1, 2).reshape(query.size(0), -1, self.num_heads * self.dim_head)
        return attn_output, attn_weights

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
        self.multihead_encoder = MultiHeadCausalAttention(projection_dim, projection_dim, projection_dim, num_heads)
        
        # Variational Components
        self.encoder_fc = nn.Linear(projection_dim, latent_dim * 2)  # Mean and log variance

        # Multi-Head Attention in Decoder
        self.multihead_decoder = MultiHeadCausalAttention(latent_dim, projection_dim, projection_dim, num_heads)
        self.decoder_projection_layer = nn.Linear(projection_dim, hidden_dim)
        
        # Graph Attention in Decoder
        self.gat_decoder = GATv2Conv(projection_dim, hidden_dim, heads=num_heads)
        
        # Fully Connected Layer to Reconstruct Input
        self.decoder_fc = nn.Linear(hidden_dim * num_heads, input_dim)

        # Hierarchical Adjacency Matrices
        self.direct_adjacency = nn.Parameter(torch.randn(num_nodes, num_nodes))  # Direct causality
        self.indirect_adjacency = nn.Parameter(torch.randn(num_nodes, num_nodes))  # Indirect causality

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

        # Decoder: Graph Attention
        x = self.gat_decoder(x, edge_index)
        x = F.relu(x)

        # Fully Connected Reconstruction
        x_reconstructed = self.decoder_fc(x)

        # Multi-Level Causal Effects
        direct_effect = torch.einsum("ij,tj->ij", self.direct_adjacency, x_reconstructed)
        indirect_effect = torch.einsum("ij,tj->ij", self.indirect_adjacency, torch.relu(x_reconstructed))
        causal_effect = direct_effect + 0.5 * indirect_effect  # Weighted combination
        
        # Apply gumbel softmax for adjacency matrix sparsification
        adj = F.gumbel_softmax(causal_effect, tau=0.5, hard=True)

        return z, mean, log_var, x_reconstructed, causal_effect, adj
