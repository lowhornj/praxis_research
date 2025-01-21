import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
import torch_geometric.transforms as T

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

class VariationalConfounder(nn.Module):
    """Variational Autoencoder for Confounder Modeling."""
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VariationalConfounder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # mean and log-variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        q_params = self.encoder(x)
        mean, log_var = torch.chunk(q_params, 2, dim=-1)
        std = torch.exp(0.5 * log_var)
        z = mean + std * torch.randn_like(std)
        x_reconstructed = self.decoder(z)
        return z, mean, log_var, x_reconstructed

class NGCWithPretrainedTKGAndTimestamps(nn.Module):
    """Neural Granger Causality Model with Pretrained TKG and Timestamp Handling."""
    def __init__(self, pretrained_tkg, input_dim, hidden_dim, output_dim, confounder_latent_dim, 
                 use_sliding_window=False, window_size=10, step_size=1, 
                 regularization_type=None, regularization_strength=0.01):
        super(NGCWithPretrainedTKGAndTimestamps, self).__init__()
        self.pretrained_tkg = pretrained_tkg
        self.gat = GraphAttentionGC(pretrained_tkg.entity_embedding.embedding_dim + input_dim, hidden_dim, output_dim)
        self.confounder = VariationalConfounder(input_dim, hidden_dim, confounder_latent_dim)

        # Parameters for sliding windows and regularization
        self.use_sliding_window = use_sliding_window
        self.window_size = window_size
        self.step_size = step_size
        self.regularization_type = regularization_type
        self.regularization_strength = regularization_strength

    def forward(self, entity_indices, relation_indices, time_series_data, edge_index, timestamp_indices):
        # Apply sliding windows if enabled
        if self.use_sliding_window:
            time_series_data = self.create_sliding_windows(time_series_data)

        # Use pretrained TKG embeddings
        entity_emb, relation_emb, _, timestamp_emb = self.pretrained_tkg(
            entity_indices, relation_indices, entity_indices, timestamp_indices
        )

        # Concatenate TKG embeddings with time-series data
        x = torch.cat([time_series_data, entity_emb, timestamp_emb], dim=-1)

        # Graph Attention for Granger Causality
        causality_scores = self.gat(x, edge_index)

        # Confounder Modeling
        z, mean, log_var, x_reconstructed = self.confounder(time_series_data)

        # Apply regularization if specified
        regularization_term = self.apply_regularization(causality_scores) if self.regularization_type else 0.0

        return causality_scores, z, mean, log_var, x_reconstructed, regularization_term

    def create_sliding_windows(self, time_series_data):
        """Create sliding windows from time-series data."""
        windows = []
        for start in range(0, len(time_series_data) - self.window_size + 1, self.step_size):
            end = start + self.window_size
            windows.append(time_series_data[start:end])
        return torch.stack(windows)

    def apply_regularization(self, causality_scores):
        """Apply L1 or L2 regularization to the causality scores."""
        if self.regularization_type == "l1":
            return self.regularization_strength * torch.sum(torch.abs(causality_scores))
        elif self.regularization_type == "l2":
            return self.regularization_strength * torch.sum(causality_scores ** 2)
        return 0.0