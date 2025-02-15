
import torch
from torch_geometric.utils import dense_to_sparse
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATv2Conv, GATConv
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch.distributions.normal import Normal
from CARAT.utils import TGCN, A3TGCN
import copy
from utils.utils import set_seed
set_seed()

# Automatically select GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device( "cpu")
print(f"Using device: {device}")

class GraphLearningModule(nn.Module):
    """
    Learns an adjacency matrix with Graph Attention.
    """
    def __init__(self, num_nodes, hidden_dim, prior_adj_matrix=None, attention_heads=4):
        super(GraphLearningModule, self).__init__()
        self.num_nodes = num_nodes
        self.attention_heads = attention_heads

        # Learnable adjacency matrix (Move to device)
        self.edge_score = nn.Parameter(torch.randn(num_nodes, num_nodes, dtype=torch.float64, device=device))

        # Prior adjacency matrix (Move to device)
        if prior_adj_matrix is not None:
            self.prior_adj = torch.tensor(prior_adj_matrix, dtype=torch.float64, device=device)
        else:
            self.prior_adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float64, device=device)

    def forward(self):
        adj_matrix = torch.sigmoid(self.edge_score) + self.prior_adj
        adj_matrix = torch.clamp(adj_matrix, 0, 1)
    
        # Convert adjacency matrix to sparse format
        edge_index, edge_weights = dense_to_sparse(adj_matrix)
    
        # Get the true number of nodes (max node index + 1)
        actual_num_nodes = edge_index.max().item() + 1
    
        # Ensure all indices are within bounds
        valid_mask = (edge_index[0] < actual_num_nodes) & (edge_index[1] < actual_num_nodes)
        edge_index = edge_index[:, valid_mask]
        edge_weights = edge_weights[valid_mask]
    
        #print(f"GraphLearningModule Output: edge_index max {edge_index.max()}, expected num_nodes {actual_num_nodes}")
    
        return edge_index.to(device).long(), edge_weights.to(device).double()
    
            

class CausalGraphVAE(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, latent_dim, num_nodes, prior_adj_matrix=None, attention_heads=4):
        super(CausalGraphVAE, self).__init__()

        # Graph Learning with Attention (Move to device)
        self.graph_learner = GraphLearningModule(num_nodes, hidden_dim, prior_adj_matrix, attention_heads).to(device)

        # Embedding layers for additional inputs
        self.entity_embed_layer = nn.Linear(embed_dim, hidden_dim, dtype=torch.float64).to(device)
        self.timestamp_embed_layer = nn.Linear(embed_dim, hidden_dim, dtype=torch.float64).to(device)

        # Temporal Graph ConvolutionalNetwork
        self.tgcn1 = A3TGCN(input_dim + 2 * hidden_dim, hidden_dim,periods=3).to(device).double()
        self.tgcn2 = A3TGCN(hidden_dim, hidden_dim,periods=3).to(device).double()

        # Latent Space
        self.mu_layer = nn.Linear(hidden_dim, latent_dim, dtype=torch.float64).to(device)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim, dtype=torch.float64).to(device)

        # Decoder with Temporal Graph ConvolutionalNetwork
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim, dtype=torch.float64).to(device)
        self.tgcn_decoder = TGCN(hidden_dim, input_dim).to(device).double()

    def encode(self, x, entity_emb, time_emb, edge_index, edge_weights):
        """
        Encoding with concatenation of entity and timestamp embeddings.
        """
        # Move inputs to correct device and convert to float64
        x = x.to(device).double()
        entity_emb = entity_emb.to(device).double()
        time_emb = time_emb.to(device).double()
        edge_index = edge_index.to(device).long()
        edge_weights = edge_weights.to(device).double()

        # Transform entity & timestamp embeddings
        entity_emb = F.relu(self.entity_embed_layer(entity_emb))
        time_emb = F.relu(self.timestamp_embed_layer(time_emb))

        # Concatenate embeddings with raw features
        x = torch.cat([x, entity_emb, time_emb], dim=-1)

        # Temporal Graph Convolutional Network Encoding (Pass edge_index dynamically)
        x = F.relu(self.tgcn1(x, edge_index, edge_weights))
       # x = F.relu(self.tgcn2(x, edge_index, edge_weights))

        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device=device, dtype=torch.float64)
        return mu + eps * std

    def decode(self, z, edge_index, edge_weights, num_nodes):
        """
        Decodes the learned representation back to the original space.
        """
        x = self.decoder_fc(z)
        x = F.relu(self.tgcn_decoder(x, edge_index, edge_weights))
        return x

    def forward(self, x, entity_emb, time_emb, num_nodes):
        edge_index, edge_weights = self.graph_learner()
        
        actual_num_nodes = x.shape[2]  # Ensure alignment
        if edge_index.max() >= actual_num_nodes:
            raise ValueError(f"Invalid edge_index detected in CausalGraphVAE! Max index: {edge_index.max()}, Expected < {actual_num_nodes}")
    
        mu, logvar = self.encode(x, entity_emb, time_emb, edge_index, edge_weights)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, edge_index, edge_weights, actual_num_nodes)  # Pass correct num_nodes
    
        return recon_x, mu, logvar, edge_weights.view(actual_num_nodes, actual_num_nodes)


def causal_vae_loss(recon_x, x, mu, logvar, adj_matrix, prior_adj_matrix, 
                    lambda_sparsity=1e-3, lambda_acyclic=1e-1, lambda_attention=1e-2):
    """
    Computes the loss function for the Causal Graph VAE with attention.
    """
    # Move all tensors to the correct device and convert to float64
    recon_x = recon_x.to(device).double()
    x = x.to(device).double()
    mu = mu.to(device).double()
    logvar = logvar.to(device).double()
    adj_matrix = adj_matrix.to(device).double()
    prior_adj_matrix = prior_adj_matrix.to(device).double()

    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL Divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Graph Sparsity loss (L1 regularization)
    sparsity_loss = lambda_sparsity * torch.norm(adj_matrix, p=1)

    # Acyclicity Constraint (Ensures DAG structure)
    eye_matrix = torch.eye(adj_matrix.shape[0], dtype=torch.float64, device=device)  # Fix: Ensure eye_matrix is float64
    H = torch.matrix_exp(adj_matrix * adj_matrix)  # Exponentiate the adjacency matrix
    acyclicity_loss = lambda_acyclic * torch.trace(H - eye_matrix)

    # Attention-based Causal Alignment Loss
    attention_loss = lambda_attention * F.mse_loss(adj_matrix, prior_adj_matrix)

    # Compute total loss
    total_loss = recon_loss + kl_loss + sparsity_loss + acyclicity_loss + attention_loss

    return total_loss


def train_causal_vae(model, optimizer, dataloader, prior_adj_matrix, num_epochs=100, patience=10):
    model.train()
    prior_adj_matrix = prior_adj_matrix.to(device).double()  # Move prior_adj_matrix to device

    best_loss = float("inf")  # Initialize best loss as infinity
    best_model_state = None  # To store best model parameters
    epochs_no_improve = 0  # Counter for early stopping
    loss_history = []  # Store epoch losses
    batch_loss_history = []  # Store all batch losses

    for epoch in range(num_epochs):
        total_loss = 0
        batch_losses = []  # Store losses per batch

        for batch_idx, (x_batch, entity_batch, time_batch) in enumerate(dataloader):
            if x_batch.shape[0]<=x_batch.shape[2]:
                continue

            optimizer.zero_grad()

            # Move data to correct device and convert to float64
            x_batch = x_batch.to(device).double()
            entity_batch = entity_batch.to(device).double()
            time_batch = time_batch.to(device).double()

            # Forward pass
            recon_x, mu, logvar, adj_matrix = model(x_batch, entity_batch, time_batch, num_nodes=x_batch.shape[1])

            # Compute loss
            loss = causal_vae_loss(
                recon_x.double(), x_batch[:,0,:], mu.double(), logvar.double(), adj_matrix.double(), prior_adj_matrix.double()
            )

            loss.backward(retain_graph=True)

            optimizer.step()

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print("NaN or Inf detected in loss function!")
                print(f"Recon loss: {recon_loss}, KL loss: {kl_loss}, Sparsity loss: {sparsity_loss}, Acyclicity loss: {acyclicity_loss}, Attention loss: {attention_loss}")
                exit(1)


            total_loss += loss.item()
            batch_losses.append(loss.item())  # Store batch loss

        avg_loss = total_loss / len(dataloader)  # Calculate average loss per epoch
        loss_history.append(avg_loss)  # Store epoch loss
        batch_loss_history.append(batch_losses)  # Store batch losses

        # Check if loss improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = copy.deepcopy(model.state_dict())  # Save best model
            epochs_no_improve = 0  # Reset counter
        else:
            epochs_no_improve += 1  # Increment counter if no improvement
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}, Best Loss = {best_loss:.4f}")

        # Early stopping condition
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch} epochs. Restoring best model.")
            model.load_state_dict(best_model_state)  # Restore best model state
            break

    return loss_history  # Return trained model, epoch losses, and batch losses