from torch.utils.data import Dataset, DataLoader
import torch
from tkgngc.embeddings import PretrainedTKGEmbeddingWithTimestamps
from tkgngc.data_processing import TKGNGCDataProcessor
from tkgngc.model import train_model
import numpy as np

def get_adjacency(cols,causal_indices,non_causal_indices,num_nodes):
    A0=torch.zeros(num_nodes,num_nodes)
    for i, row in enumerate(A0):
        for j, column in enumerate(row):
            if (j in non_causal_indices) and (i in causal_indices) & (i!=j):
                A0[i,j] = 0.5
    return A0

class create_granger_gat_data:
    def __init__(self,pretrained_tkg,data_class):
        self.edge_index = data_class.edge_index
        self.entity_indices = data_class.entity_indices
        self.relation_indices = data_class.relation_indices
        self.timestamp_indices = data_class.timestamp_indices
        self.time_series_data = data_class.time_series_tensor
        self.pretrained_tkg = pretrained_tkg

    def retrain_tkg(self):
        quads = (
            self.entity_indices[:-1],  # Head entities
            self.relation_indices,  # Relations
            self.entity_indices[1:],  # Tail entities (shifted example)
            self.timestamp_indices[:-1],  # Timestamps
        )
        self.pretrained_tkg.pretrain(quads, learning_rate=0.01, epochs=10)

        self.entity_emb, self.relation_emb, _, self.timestamp_emb = self.pretrained_tkg(
        self.entity_indices, self.relation_indices, self.entity_indices, self.timestamp_indices
        )

        self.enriched_features = torch.cat([self.time_series_data, self.entity_emb, self.timestamp_emb], dim=-1)

def create_lagged_features(data, num_lags, pad_value=np.nan):
    """
    Creates lagged versions of input data.
    
    Args:
        data (numpy.ndarray or torch.Tensor): Input data of shape (rows, features).
        num_lags (int): Number of lagged time steps to create.
        pad_value (float, optional): Value to use for padding. Defaults to NaN.
        
    Returns:
        torch.Tensor: Lagged data of shape (rows, num_lags, features).
    """
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)  # Convert to PyTorch tensor if needed
    
    rows, features = data.shape
    lagged_data = torch.full((rows, num_lags, features), pad_value, dtype=data.dtype)  # Initialize with pad_value
    
    for lag in range(1, num_lags + 1):
        lagged_data[lag:, lag - 1, :] = data[:-lag]  # Shift data down by `lag` steps
    
    return lagged_data

class TimeSeriesDataset(Dataset):
    """
    A generic PyTorch Dataset class for unsupervised learning that includes:
    - Features
    - Entity embeddings
    - Time embeddings

    Args:
        data (numpy.ndarray or torch.Tensor): Input features of shape (rows, features).
        entity_embeddings (numpy.ndarray or torch.Tensor): Entity embeddings of shape (rows, embed_dim).
        time_embeddings (numpy.ndarray or torch.Tensor): Time embeddings of shape (rows, embed_dim).
    """

    def __init__(self, data, entity_embeddings, time_embeddings):
        # Convert inputs to PyTorch tensors if they are numpy arrays
        self.data = torch.tensor(data, dtype=torch.float32) if isinstance(data, np.ndarray) else data
        self.entity_embeddings = torch.tensor(entity_embeddings, dtype=torch.float32) if isinstance(entity_embeddings, np.ndarray) else entity_embeddings
        self.time_embeddings = torch.tensor(time_embeddings, dtype=torch.float32) if isinstance(time_embeddings, np.ndarray) else time_embeddings

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a tuple: (features, entity embeddings, time embeddings).
        """
        return self.data[idx], self.entity_embeddings[idx], self.time_embeddings[idx]

