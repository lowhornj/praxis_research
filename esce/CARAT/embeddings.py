import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
import torch_geometric.transforms as T

import torch
import torch.nn as nn
import torch.optim as optim


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

    def pretrain(self, quads, learning_rate=0.01, epochs=100, patience=10, verbose=True):
        """
        Pretrain embeddings using knowledge graph quads (with timestamps) with early stopping.

        Args:
            quads (torch.Tensor): Tensor of shape [num_samples, 4] containing (head, relation, tail, timestamp).
            learning_rate (float): Learning rate for the optimizer.
            epochs (int): Maximum number of epochs.
            patience (int): Number of epochs to wait without improvement before stopping.
            verbose (bool): Whether to print progress.
        """
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        best_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(epochs):
            optimizer.zero_grad()

            head, relation, tail, timestamp = quads[ 0], quads[ 1], quads[ 2], quads[ 3]

            # Embed the inputs
            head_emb = self.entity_embedding(head)
            relation_emb = self.relation_embedding(relation)
            tail_emb = self.entity_embedding(tail)
            timestamp_emb = self.timestamp_embedding(timestamp)

            # Predict tail embedding using a TransE-style scoring function
            predicted_tail_emb = head_emb + relation_emb + timestamp_emb

            # Compute loss (distance between predicted and actual tail embedding)
            loss = loss_fn(predicted_tail_emb, tail_emb)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            # Early stopping logic
            if loss.item() < best_loss:
                best_loss = loss.item()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

            if epochs_without_improvement >= patience:
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


