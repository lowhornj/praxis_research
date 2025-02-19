import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
import torch_geometric.transforms as T

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader



class PretrainedTKGEmbeddingWithTimestamps(nn.Module):
    """
    Pretrained Temporal Knowledge Graph Embedding Module with Timestamps.
    Designed for learning temporal dependencies between variables in time series data.
    """

    def __init__(self, num_entities, num_relations, embedding_dim, num_timestamps):
        """
        Args:
            num_entities (int): Number of unique entities (variables in the time series).
            num_relations (int): Number of unique relations (e.g., lags).
            embedding_dim (int): Size of each embedding vector.
            num_timestamps (int): Number of discrete time bins.
        """
        super().__init__()
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
        self.timestamp_embedding = nn.Embedding(num_timestamps, embedding_dim)

    def pretrain(self, quads, batch_size=1024, learning_rate=0.01, epochs=100, patience=10, verbose=True):
        """
        Pretrain embeddings using knowledge graph quads (with timestamps) with mini-batching and early stopping.
    
        Args:
            quads (tuple): Tuple of tensors (head, relation, tail, timestamp).
            batch_size (int): Mini-batch size.
            learning_rate (float): Learning rate for optimizer.
            epochs (int): Maximum number of epochs.
            patience (int): Early stopping patience.
            verbose (bool): Print progress.
        """
        head, relation, tail, timestamp = quads
    
        # Wrap data into a TensorDataset and DataLoader for batching
        dataset = TensorDataset(head, relation, tail, timestamp)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
    
        best_loss = float('inf')
        epochs_without_improvement = 0
    
        for epoch in range(epochs):
            epoch_loss = 0.0
    
            for batch in dataloader:
                batch_head, batch_relation, batch_tail, batch_timestamp = batch
    
                optimizer.zero_grad()
    
                # Embed batch inputs
                head_emb = self.entity_embedding(batch_head)
                relation_emb = self.relation_embedding(batch_relation)
                tail_emb = self.entity_embedding(batch_tail)
                timestamp_emb = self.timestamp_embedding(batch_timestamp)
    
                # TransE-style prediction
                predicted_tail_emb = head_emb + relation_emb + timestamp_emb
    
                # Loss calculation (distance to true tail embedding)
                loss = loss_fn(predicted_tail_emb, tail_emb)
    
                loss.backward()
                optimizer.step()
    
                epoch_loss += loss.item()
    
            epoch_loss /= len(dataloader)
    
            if verbose:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
    
            # Early stopping check
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
    
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch}. Best loss: {best_loss:.4f}")
                break

    def forward(self, head, relation, tail, timestamp):
        """
        Forward pass to retrieve embeddings.

        Args:
            head, relation, tail, timestamp (torch.Tensor): Tensor inputs.

        Returns:
            Embeddings for head, relation, tail, and timestamp.
        """
        head_emb = self.entity_embedding(head)
        relation_emb = self.relation_embedding(relation)
        tail_emb = self.entity_embedding(tail)
        timestamp_emb = self.timestamp_embedding(timestamp)
        return head_emb, relation_emb, tail_emb, timestamp_emb