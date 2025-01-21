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

    def pretrain(self, quads, learning_rate=0.01, epochs=100):
        """Pretrain embeddings using knowledge graph quads (with timestamps)."""
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            optimizer.zero_grad()

            head, relation, tail, timestamp = quads
            head_emb = self.entity_embedding(head)
            relation_emb = self.relation_embedding(relation)
            tail_emb = self.entity_embedding(tail)
            timestamp_emb = self.timestamp_embedding(timestamp)

            predicted_tail_emb = head_emb + relation_emb + timestamp_emb
            loss = loss_fn(predicted_tail_emb, tail_emb)

            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

    def forward(self, head, relation, tail, timestamp):
        head_emb = self.entity_embedding(head)
        relation_emb = self.relation_embedding(relation)
        tail_emb = self.entity_embedding(tail)
        timestamp_emb = self.timestamp_embedding(timestamp)
        return head_emb, relation_emb, tail_emb, timestamp_emb
