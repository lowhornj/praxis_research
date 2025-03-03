
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
    Learns intra-slice (W) and inter-slice (A) adjacency matrices separately.
    """
    def __init__(self, num_nodes, hidden_dim, prior_adj_matrix=None):
        super(GraphLearningModule, self).__init__()
        self.num_nodes = num_nodes
        
        # Learnable adjacency matrices
        self.W_score = nn.Parameter(torch.randn(num_nodes, num_nodes, dtype=torch.float64, device=device))
        self.A_score = nn.Parameter(torch.randn(num_nodes, num_nodes, dtype=torch.float64, device=device))

        # Prior adjacency matrix (if any)
        if prior_adj_matrix is not None:
            self.prior_adj = torch.tensor(prior_adj_matrix, dtype=torch.float64, device=device)
        else:
            self.prior_adj = None

    def forward(self):
        # Sigmoid activation to constrain values between 0 and 1
        W = torch.sigmoid(self.W_score)
        A = torch.sigmoid(self.A_score)

        # Apply NoTears constraint to W (intra-slice)
        W = torch.clamp(W, 0, 1)
        A = torch.clamp(A, 0, 1)

        # Ensure no self-cycles
        W.fill_diagonal_(0)

        return W, A  # Return separate adjacency matrices



class GraphAttentionLearningModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes, heads=4):
        super(GraphAttentionLearningModule, self).__init__()
        self.num_nodes = num_nodes
        self.input_emb = nn.Parameter(torch.randn(num_nodes, input_dim,device=device))
        self.conv = GATConv(input_dim, hidden_dim, heads=heads, concat=True, add_self_loops=False).to(device)

    def forward(self):
        edge_index = torch.cartesian_prod(torch.arange(self.num_nodes), torch.arange(self.num_nodes)).T
        edge_index = edge_index.to(device)
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]

        node_embeddings, (attention_edge_index,attention_weights) = self.conv(self.input_emb, edge_index,return_attention_weights=True)

        if attention_weights.dim() == 2:
            attention_weights = attention_weights.mean(dim=1)

        # Convert edge attention back to dense adjacency matrix
        adj_matrix = torch.zeros(self.num_nodes, self.num_nodes, device=device)
        #adj_matrix[edge_index[0], edge_index[1]] = attention_weights
        adj_matrix = adj_matrix.index_put(
            (attention_edge_index[0], attention_edge_index[1]),
            attention_weights,
            accumulate=True
        )

        return edge_index, adj_matrix
    
            

class CausalGraphVAE(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, latent_dim, num_nodes, prior_adj_matrix=None, attention_heads=4):
        super(CausalGraphVAE, self).__init__()

        self.register_buffer('alpha', torch.tensor(0.0, dtype=torch.float64, device=device))
        self.register_buffer('rho', torch.tensor(1.0, dtype=torch.float64, device=device))


        # Graph Learning with Attention (Move to device)
        self.graph_learner = GraphLearningModule(num_nodes, hidden_dim, prior_adj_matrix).to(device)
        #self.graph_learner = GraphAttentionLearningModule(input_dim,hidden_dim,num_nodes,heads=4)

        # Embedding layers for additional inputs
        self.entity_embed_layer = nn.Linear(embed_dim, hidden_dim, dtype=torch.float64).to(device)
        self.timestamp_embed_layer = nn.Linear(embed_dim, hidden_dim, dtype=torch.float64).to(device)

        # Temporal Graph ConvolutionalNetwork
        self.tgcn1 = A3TGCN(input_dim + 2 * hidden_dim, hidden_dim,periods=3).to(device).double()
        #self.tgcn2 = A3TGCN(hidden_dim, hidden_dim,periods=3).to(device).double()

        # Latent Space
        self.mu_layer = nn.Linear(hidden_dim, latent_dim, dtype=torch.float64).to(device)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim, dtype=torch.float64).to(device)

        # Decoder with Temporal Graph ConvolutionalNetwork
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim, dtype=torch.float64).to(device)
        self.tgcn_decoder = TGCN(hidden_dim, input_dim).to(device).double()

    def encode(self, x, entity_emb, time_emb, edge_index, edge_weights):
        """
        Encoding with concatenation of entity and timestamp embeddings.
        Uses a single merged edge index and weight.
        """
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
    
        # Temporal Graph Convolutional Network Encoding (Pass merged edge_index)
        x = F.relu(self.tgcn1(x, edge_index, edge_weights))
    
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
        Uses a single merged edge index and weight.
        """
        x = self.decoder_fc(z)  # Linear transformation to latent space
    
        # Apply relu activation
        x = F.relu(self.tgcn_decoder(x, edge_index, edge_weights))
    
        return x



    def forward(self, x, entity_emb, time_emb, num_nodes):
        W, A = self.graph_learner()  # Get W (intra-slice) and A (inter-slice)
        
        # Convert to sparse format
        edge_index_W, edge_weights_W = dense_to_sparse(W)
        edge_index_A, edge_weights_A = dense_to_sparse(A)
    
        actual_num_nodes = x.shape[2]  # Ensure alignment
        if edge_index_W.max() >= actual_num_nodes or edge_index_A.max() >= actual_num_nodes:
            raise ValueError(f"Invalid edge_index detected! Max index: {max(edge_index_W.max(), edge_index_A.max())}, Expected < {actual_num_nodes}")
    
        # Merge intra-slice (W) and inter-slice (A) edges
        edge_index = torch.cat([edge_index_W, edge_index_A], dim=1)
        edge_weights = torch.cat([edge_weights_W, edge_weights_A])
    
        mu, logvar = self.encode(x, entity_emb, time_emb, edge_index, edge_weights)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, edge_index, edge_weights, actual_num_nodes)  # Pass correct num_nodes
    
        return recon_x, mu, logvar, W, A  # Return adjacency matrices for debugging
    


def notears_constraint(W):
    """
    NoTears function h(W) = trace(expm(W * W)) - d
    Ensures acyclicity in the contemporaneous (intra-slice) adjacency matrix.
    """
    d = W.shape[0]
    WW = W * W  # Element-wise Hadamard product
    expm_ww = torch.matrix_exp(WW)  # Matrix exponential
    return torch.trace(expm_ww) - d  # NoTears condition



def augmented_lagrangian_loss(
    recon_x, x, mu, logvar, W, model, 
    lambda_sparsity=1e-3, 
    lambda_attention=1e-2
):
    """
    W is your adjacency matrix. model.alpha and model.rho 
    are your Lagrange multiplier and penalty parameter.
    """
    # 1) Main loss: reconstruction + KL
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    main_loss = recon_loss + kl_loss
    
    # 2) Additional penalty terms you want
    # e.g. L1 (sparsity):
    sparsity_loss = lambda_sparsity * torch.norm(W, p=1)
    
    # e.g. Attention alignment with prior:
    #   (assuming the user wants a standard MSE penalty vs prior)
    #   prior_adj = ...
    #   attention_loss = lambda_attention * F.mse_loss(W, prior_adj)
    #   ...
    
    # 3) Compute the NOTEARS constraint
    h_val = notears_constraint(W)
    
    # 4) The augmented Lagrangian piece: alpha * h_val + 0.5 * rho * (h_val**2)
    lagrangian_term = model.alpha * h_val + 0.5 * model.rho * (h_val * h_val)
    
    # 5) Combine everything
    total_loss = main_loss + sparsity_loss + lagrangian_term
    return total_loss, h_val




def causal_vae_loss(recon_x, x, mu, logvar, W, A, lambda_sparsity=1e-3, lambda_acyclic=1e-1):
    """
    Loss function enforcing acyclicity on intra-slice edges (W) but not time-lagged edges (A).
    """
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # L1 sparsity regularization
    sparsity_loss = lambda_sparsity * (torch.norm(W, p=1) + torch.norm(A, p=1))

    # Acyclicity constraint ONLY on intra-slice edges
    h_val = notears_constraint(W)
    lagrangian_term = model.alpha * h_val + 0.5 * model.rho * (h_val * h_val)


    total_loss = recon_loss + kl_loss + sparsity_loss + lagrangian_term
    return total_loss, h_val



def train_causal_vae(model, optimizer, dataloader, prior_adj_matrix, num_epochs=100, patience=20):
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
            recon_x, mu, logvar, W, A = model(x_batch, entity_batch, time_batch, num_nodes=x_batch.shape[1])



            # Compute loss
            """loss = causal_vae_loss(
                recon_x.double(), x_batch[:,0,:], mu.double(), logvar.double(), adj_matrix.double(), prior_adj_matrix.double()
            )"""

            loss, h_val = augmented_lagrangian_loss(
                recon_x.double(),
                x_batch[:,0,:].double(),
                mu.double(),
                logvar.double(),
                W.double(),
                model,
                lambda_sparsity=1e-3
            )


            loss.backward(retain_graph=True)

            optimizer.step()

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print("NaN or Inf detected in loss function!")
                print(f"Recon loss: {recon_loss}, KL loss: {kl_loss}, Sparsity loss: {sparsity_loss}, Acyclicity loss: {acyclicity_loss}, Attention loss: {attention_loss}")
                exit(1)


            total_loss += loss.item()
            batch_losses.append(loss.item())  # Store batch loss
        
        with torch.no_grad():
            model.alpha += model.rho * h_val.item()
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