from torch.utils.data import Dataset, DataLoader
import torch
from tkgngc.embeddings import PretrainedTKGEmbeddingWithTimestamps
from tkgngc.data_processing import TKGNGCDataProcessor
from tkgngc.model import train_model

def get_adjacency(cols,causal_indices,non_causal_indices,num_nodes):
    A0=torch.zeros(num_nodes,num_nodes)
    for i, row in enumerate(A0):
        for j, column in enumerate(row):
            if (j in non_causal_indices) and (i in causal_indices) & (i!=j):
                A0[i,j] = 0.75
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


class GraphDataset(Dataset):
    def __init__(self, x, entity_emb, time_emb):
        """
        Initializes dataset with feature data, entity embeddings, and timestamp embeddings.
        """
        self.x = torch.tensor(x, dtype=torch.float)  # Shape: (num_samples, num_features)
        self.entity_emb = torch.tensor(entity_emb, dtype=torch.float)  # Shape: (num_samples, embed_dim)
        self.time_emb = torch.tensor(time_emb, dtype=torch.float)  # Shape: (num_samples, embed_dim)

    def __len__(self):
        return len(self.x)  # Number of samples

    def __getitem__(self, idx):
        """
        Retrieves one sample from the dataset.
        """
        return self.x[idx], self.entity_emb[idx], self.time_emb[idx]
