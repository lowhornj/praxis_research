import torch
from torch_geometric.utils import dense_to_sparse
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATv2Conv,GATConv
from torch_geometric.data import Data
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch.distributions.normal import Normal
from tkgngc.utils import vae_loss, gumbel_softmax
import pandas as pd
from utils.utils import set_seed
import numpy as np
from CARAT.utils import *
from tkgngc.embeddings import PretrainedTKGEmbeddingWithTimestamps
from tkgngc.data_processing import TKGNGCDataProcessor
from tkgngc.model import train_model

class GraphLearningModule(nn.Module):
    """
    Learns an adjacency matrix with Graph Attention.
    """
    def __init__(self, num_nodes, hidden_dim, prior_adj_matrix=None, attention_heads=4):
        super(GraphLearningModule, self).__init__()
        self.num_nodes = num_nodes
        self.attention_heads = attention_heads

        # Learnable adjacency matrix
        self.edge_score = nn.Parameter(torch.randn(num_nodes, num_nodes))

        # Prior adjacency matrix
        if prior_adj_matrix is not None:
            self.prior_adj = torch.tensor(prior_adj_matrix, dtype=torch.float)
        else:
            self.prior_adj = torch.zeros(num_nodes, num_nodes)

    def forward(self, x):
        """
        Compute the learned adjacency matrix with attention.
        """
        adj_matrix = torch.sigmoid(self.edge_score) + self.prior_adj
        adj_matrix = torch.clamp(adj_matrix, 0, 1)

        # Convert to sparse edge list
        edge_index, edge_weights = dense_to_sparse(adj_matrix)

        return edge_index, edge_weights 

class CausalGraphVAE(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, latent_dim, num_nodes, prior_adj_matrix=None, attention_heads=4):
        super(CausalGraphVAE, self).__init__()

        # Graph Learning with Attention
        self.graph_learner = GraphLearningModule(num_nodes, hidden_dim, prior_adj_matrix, attention_heads)

        # Embedding layers for additional inputs
        self.entity_embed_layer = nn.Linear(embed_dim, hidden_dim)  # Entity embeddings
        self.timestamp_embed_layer = nn.Linear(embed_dim, hidden_dim)  # Timestamp embeddings

        # Temporal Graph ConvolutionalNetwork
        self.tgcn1 = TGCN(input_dim + 2 * hidden_dim, hidden_dim)
        self.tgcn2 = TGCN(hidden_dim, hidden_dim)

        # Latent Space
        self.mu_layer = nn.Linear(hidden_dim , latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim , latent_dim)

        # Decoder with Temporal Graph ConvolutionalNetwork
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim )
        self.tgcn_decoder = TGCN(hidden_dim, input_dim)

    def encode(self, x, entity_emb, time_emb, edge_index, edge_weights):
        """
        Encoding with concatenation of entity and timestamp embeddings.
        """
        # Transform entity & timestamp embeddings to match hidden space
        entity_emb = F.relu(self.entity_embed_layer(entity_emb))
        time_emb = F.relu(self.timestamp_embed_layer(time_emb))

        # Concatenate embeddings with raw features
        x = torch.cat([x, entity_emb, time_emb], dim=-1)

        # Temporal Graph Convolutional Network Encoding (Pass edge_index dynamically)
        x = F.relu(self.tgcn1(x, edge_index, edge_weights))
        x = F.relu(self.tgcn2(x, edge_index, edge_weights))
        #x = x.view(x.shape[0], -1)  # Flatten
        #x = x.squeeze(1)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, edge_index, edge_weights, num_nodes):
        """
        Decodes the learned representation back to the original space.
        """
        x = self.decoder_fc(z)
        #x = x.view(num_nodes, -1)
        x = F.relu(self.tgcn_decoder(x, edge_index, edge_weights))
        return x

    def forward(self, x, entity_emb, time_emb, num_nodes):
        """
        Forward pass with graph learning and encoding.
        """
        edge_index, edge_weights = self.graph_learner()  # Learn adjacency matrix and edges
        mu, logvar = self.encode(x, entity_emb, time_emb, edge_index, edge_weights)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, edge_index, edge_weights, num_nodes)
        return recon_x, mu, logvar, edge_weights.view(num_nodes,num_nodes)


def causal_vae_loss(recon_x, x, mu, logvar, adj_matrix, prior_adj_matrix, lambda_sparsity=1e-3, lambda_acyclic=1e-2, lambda_attention=1e-2):
    """
    Computes the loss function for the Causal Graph VAE with attention.
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL Divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Graph Sparsity loss (L1 regularization)
    sparsity_loss = lambda_sparsity * torch.norm(adj_matrix, p=1)

    # Acyclicity Constraint (Ensures DAG structure)
    H = torch.matrix_exp(adj_matrix * adj_matrix)  
    acyclicity_loss = lambda_acyclic * torch.trace(H - torch.eye(adj_matrix.shape[0]))

    # Attention-based Causal Alignment Loss
    attention_loss = lambda_attention * F.mse_loss(adj_matrix, prior_adj_matrix)

    return recon_loss + kl_loss + sparsity_loss + acyclicity_loss + attention_loss


def train_causal_vae(model, optimizer, dataloader, prior_adj_matrix,num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (x_batch,entity_batch,time_batch) in enumerate(dataloader):
            optimizer.zero_grad()
            recon_x, mu, logvar, adj_matrix = model(x_batch, entity_batch, time_batch, num_nodes=x_batch.shape[1])
            loss = causal_vae_loss(recon_x, x_batch, mu, logvar, adj_matrix.to(torch.float), prior_adj_matrix.to(torch.float))
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
