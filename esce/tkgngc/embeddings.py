import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
import torch_geometric.transforms as T

class PretrainedTKGEmbeddingWithTimestamps(nn.Module):
    """Pretrained Temporal Knowledge Graph Embedding Module with Timestamps."""
    def __init__(self, num_entities, num_relations, embedding_dim, num_timestamps):
        super(PretrainedTKGEmbeddingWithTimestamps, self).__init__()
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
        self.timestamp_embedding = nn.Embedding(num_timestamps, embedding_dim)

    def pretrain(self, quads, learning_rate=0.01, epochs=100, patience=10, verbose=True):
        """
        Pretrain embeddings using knowledge graph quads (with timestamps) with early stopping.
        
        Args:
            quads (tuple): Tuple of tensors (head, relation, tail, timestamp).
            learning_rate (float): Learning rate for the optimizer.
            epochs (int): Maximum number of epochs.
            patience (int): Number of epochs to wait without improvement before stopping.
            verbose (bool): Whether to print progress.
        """
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
    
        # Validate input sizes
        head, relation, tail, timestamp = quads
        assert head.size(0) == relation.size(0) == tail.size(0) == timestamp.size(0), \
            "All tensors in quads must have the same size in dimension 0."
    
        best_loss = float('inf')
        epochs_without_improvement = 0
    
        for epoch in range(epochs):
            optimizer.zero_grad()
    
            # Embed the inputs
            head_emb = self.entity_embedding(head)
            relation_emb = self.relation_embedding(relation)
            tail_emb = self.entity_embedding(tail)
            timestamp_emb = self.timestamp_embedding(timestamp)
    
            # Predict tail embedding
            predicted_tail_emb = head_emb + relation_emb + timestamp_emb
    
            # Compute loss
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
        head_emb = self.entity_embedding(head)
        relation_emb = self.relation_embedding(relation)
        tail_emb = self.entity_embedding(tail)
        timestamp_emb = self.timestamp_embedding(timestamp)
        return head_emb, relation_emb, tail_emb, timestamp_emb
