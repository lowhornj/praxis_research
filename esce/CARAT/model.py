
import torch
from torch_geometric.utils import dense_to_sparse
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATv2Conv, GATConv
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch.distributions.normal import Normal
from CARAT.utils import TGCN, A3TGCN, ARMAConv
import copy
import os
from utils.utils import set_seed
import numpy as np
os.environ['CUBLAS_WORKSPACE_CONFIG'] = '4096:8'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = "1"
torch.backends.cudnn.benchmark=False
torch.autograd.profiler.profile(enabled=False)
torch.autograd.profiler.emit_nvtx(enabled=False)
torch.autograd.set_detect_anomaly(mode=False)

set_seed()

# Automatically select GPU if available, otherwise use CPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = torch.device( "cpu")
print(f"Using device: {device}")

def linear_annealing(epoch, start_beta, end_beta, total_epochs):

    beta = start_beta + (end_beta - start_beta) * epoch / total_epochs

    return beta

#https://github.com/haofuml/cyclical_annealing/tree/master
def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L 

def replace_zero(tensor, small_number=1e-8):
  """
  Replaces zeros in a PyTorch tensor with a small number.

  Args:
    tensor: The input tensor.
    small_number: The small number to replace zeros with (default: 1e-8).

  Returns:
    A new tensor with zeros replaced by the small number.
  """
  return torch.where(tensor == 0, torch.tensor(small_number, dtype=tensor.dtype), tensor)

class GraphLearningModule(nn.Module):
    """
    Learns intra-slice (W) and inter-slice (A) adjacency matrices separately.
    """
    def __init__(self, num_nodes, hidden_dim, prior_adj_matrix=None):
        super(GraphLearningModule, self).__init__()
        self.num_nodes = num_nodes
        
        # Learnable adjacency matrices
        self.W_score = nn.Parameter(torch.randn(num_nodes, num_nodes, dtype=torch.float32, device=device))
        self.A_score = nn.Parameter(torch.randn(num_nodes, num_nodes, dtype=torch.float32, device=device))

        # Prior adjacency matrix (if any)
        if prior_adj_matrix is not None:
            self.prior_adj = torch.tensor(prior_adj_matrix, dtype=torch.float32, device=device)
        else:
            self.prior_adj = None

    def forward(self):
        # Sigmoid activation to constrain values between 0 and 1

        # Apply NoTears constraint to W (intra-slice)
        W = torch.clamp(self.W_score, 0, 1)
        A = torch.clamp(self.A_score, 0, 1)

        # Ensure no self-cycles
        # Use small numbers to represent zeros to avoid issues with dense to sparse representations
        W = replace_zero(W.fill_diagonal_(0))
        A = replace_zero(A.fill_diagonal_(0))

        return F.softmax(W), F.softmax(A)  # Return separate adjacency matrices


class DynamicGraphAttention(nn.Module):
    """Learns a time-dependent causal graph with attention."""
    def __init__(self, hidden_dim,num_nodes,prior_adj_matrix):
        super().__init__()
        self.prior_adj = prior_adj_matrix
        self.proj_start = nn.Linear(num_nodes,hidden_dim, device=device)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, device=device)
        self.A_score = torch.randn(num_nodes, num_nodes, device=device)
        self.proj_end =  nn.Linear(hidden_dim,num_nodes, device=device)

    def forward(self, x):
        adj_dynamic = self.proj_start(x)
        adj_dynamic, _ = self.attn(adj_dynamic, adj_dynamic, adj_dynamic)
        adj_dynamic = self.proj_end(adj_dynamic)
        W = torch.einsum('bik,bij->kj', adj_dynamic, adj_dynamic)
        W = replace_zero(W.fill_diagonal_(0))
        if self.prior_adj is not None:
            # Add prior beliefs
            W = W + self.prior_adj
        A = replace_zero(self.A_score.fill_diagonal_(0))
        
        return torch.clamp(W,0,1), torch.clamp(A,0,1)


class ConfounderVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, z
     

class CausalGraphVAE(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim1,hidden_dim2, latent_dim, num_nodes, prior_adj_matrix=None,n_lags=3, attention_heads=4, use_embeddings=True):
        super(CausalGraphVAE, self).__init__()

        self.register_buffer('alpha', torch.tensor(0.0, dtype=torch.float32, device=device))
        self.register_buffer('rho', torch.tensor(1.0, dtype=torch.float32, device=device))
        self.use_embeddings = use_embeddings
        self.confounder_vae = ConfounderVAE(input_dim, latent_dim).to(device)
        # Graph Learning with Attention (Move to device)
        self.graph_learner = DynamicGraphAttention(num_nodes=num_nodes, hidden_dim=hidden_dim1, prior_adj_matrix=prior_adj_matrix).to(device)
        #self.graph_learner = GraphAttentionLearningModule(input_dim,hidden_dim,num_nodes,heads=4)
        if self.use_embeddings:
            # Embedding layers for additional inputs
            self.entity_embed_layer = nn.Linear(embed_dim, hidden_dim1, dtype=torch.float32).to(device)
            self.timestamp_embed_layer = nn.Linear(embed_dim, hidden_dim1, dtype=torch.float32).to(device)
            # Temporal Graph ConvolutionalNetwork
            self.tgcn1 = A3TGCN(input_dim + 2 * hidden_dim1, hidden_dim1,periods=n_lags).to(device).float()
        else:
            self.tgcn1 = A3TGCN(input_dim, hidden_dim1,periods=3).to(device).float()

        #self.tgcn2 = A3TGCN(hidden_dim, hidden_dim,periods=3).to(device).float()

        self.arma1 = ARMAConv(hidden_dim1, hidden_dim2,num_layers = 3).to(device).float()

        # Latent Space
        self.mu_layer = nn.Linear(hidden_dim2, latent_dim, dtype=torch.float32).to(device)
        self.logvar_layer = nn.Linear(hidden_dim2, latent_dim, dtype=torch.float32).to(device)

        # Decoder with Temporal Graph ConvolutionalNetwork
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim2, dtype=torch.float32).to(device)
        self.arma2 = ARMAConv(hidden_dim2, hidden_dim1,num_layers = 3).to(device).float()
        self.tgcn_decoder = TGCN(hidden_dim1, input_dim).to(device).float()

    def encode(self, x, entity_emb, time_emb, edge_index, edge_weights):
        """
        Encoding with concatenation of entity and timestamp embeddings.
        Uses a single merged edge index and weight.
        """
        x = x.to(device).float()
        if self.use_embeddings:
            entity_emb = entity_emb.to(device).float()
            time_emb = time_emb.to(device).float()
            # Transform entity & timestamp embeddings
            entity_emb = F.relu(self.entity_embed_layer(entity_emb))
            time_emb = F.relu(self.timestamp_embed_layer(time_emb))

                    # Concatenate embeddings with raw features
            x = torch.cat([x, entity_emb, time_emb], dim=-1)
                
        edge_index = edge_index.to(device).long()
        edge_weights = edge_weights.to(device).float()

        
        # Temporal Graph Convolutional Network Encoding (Pass merged edge_index)
        x = F.relu(self.tgcn1(x, edge_index, edge_weights))

        x = self.arma1(x,edge_index,edge_weights)
    
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device=device, dtype=torch.float32)
        return mu + eps * std

    def decode(self, z, edge_index, edge_weights, num_nodes):
        """
        Decodes the learned representation back to the original space.
        Uses a single merged edge index and weight.
        """
        x = self.decoder_fc(z)  # Linear transformation to latent space
        x = self.arma2(x,edge_index,edge_weights)
        # Apply relu activation
        x = F.relu(self.tgcn_decoder(x, edge_index, edge_weights))
    
        return x

    def forward(self, x, entity_emb, time_emb, num_nodes):
        W, A = self.graph_learner(x)  # Get W (intra-slice) and A (inter-slice)
        
        # Convert to sparse format
        edge_index_W, edge_weights_W = dense_to_sparse(W)
        edge_index_A, edge_weights_A = dense_to_sparse(A)
    
        actual_num_nodes = x.shape[2]  # Ensure alignment
        if edge_index_W.max() >= actual_num_nodes or edge_index_A.max() >= actual_num_nodes:
            raise ValueError(f"Invalid edge_index detected! Max index: {max(edge_index_W.max(), edge_index_A.max())}, Expected < {actual_num_nodes}")
    
        # Merge intra-slice (W) and inter-slice (A) edges
        edge_index = edge_index_W
        edge_weights = edge_weights_W
        #edge_index = torch.cat([edge_index_W, edge_index_A], dim=1)
        #edge_weights = torch.cat([edge_weights_W, edge_weights_A])
        confounder_x, latent_confounders = self.confounder_vae(x)
        mu, logvar = self.encode(confounder_x, entity_emb, time_emb, edge_index, edge_weights)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, edge_index, edge_weights, actual_num_nodes)  # Pass correct num_nodes
    
        return recon_x, mu, logvar, W, A  # Return adjacency matrices for debugging
    


def notears_constraint(W):
    """
    NoTears function h(W) = trace(expm(W * W)) - d
    Ensures acyclicity in the contemporaneous (intra-slice) adjacency matrix.
    """
    d = W.shape[0]
    W = torch.nn.functional.softmax(W,dim=1)
    WW = W * W  # Element-wise Hadamard product
    expm_ww = torch.matrix_exp(WW)  # Matrix exponential
    return torch.trace(expm_ww) - d  # NoTears condition



def augmented_lagrangian_loss(
    recon_x, x, mu, logvar, W, model, 
    beta,
    tau,
    lambda_sparsity=1e-3, 

):
    """
    W is your adjacency matrix. model.alpha and model.rho 
    are your Lagrange multiplier and penalty parameter.
    """
    # 1) Main loss: reconstruction + KL
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * beta
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
    lagrangian_term = (model.alpha * h_val + 0.5 * model.rho * (h_val * h_val)) #* tau
    
    # 5) Combine everything
    total_loss = main_loss + sparsity_loss + lagrangian_term
    return recon_loss, kl_loss, sparsity_loss, lagrangian_term, h_val

def train_causal_vae(model, optimizer, dataloader, prior_adj_matrix, num_epochs=100, patience=20,
                     use_embeddings=False,lin_anneal_start=1e-8,lin_anneal_end=1e-1,lambda_sparsity=1e-3
                    ):
    model.train()
    prior_adj_matrix = prior_adj_matrix.to(device).float()  # Move prior_adj_matrix to device

    best_loss = float("inf")  # Initialize best loss as infinity
    best_model_state = None  # To store best model parameters
    epochs_no_improve = 0  # Counter for early stopping
    loss_history = []  # Store epoch losses
    batch_loss_history = []  # Store all batch losses
    beta_schedule = frange_cycle_linear(num_epochs)

    for epoch in range(num_epochs):
        tau = linear_annealing(epoch,lin_anneal_start,lin_anneal_end,num_epochs)
        beta = beta_schedule[epoch]
        total_loss = 0
        batch_losses = []  # Store losses per batch

        if use_embeddings:
            for batch_idx, (x_batch, entity_batch, time_batch) in enumerate(dataloader):
                if x_batch.shape[0]<=x_batch.shape[2]:
                    continue
    
                optimizer.zero_grad()
    
                # Move data to correct device and convert to float32
                x_batch = x_batch.to(device).float()
                entity_batch = entity_batch.to(device).float()
                time_batch = time_batch.to(device).float()
    
                # Forward pass
                recon_x, mu, logvar, W, A = model(x_batch, entity_batch, time_batch, num_nodes=x_batch.shape[1])
    
                recon_loss, kl_loss, sparsity_loss, lagrangian_term, h_val = augmented_lagrangian_loss(
                    recon_x.float(),
                    x_batch[:,0,:].float(),
                    mu.float(),
                    logvar.float(),
                    W.float(),
                    model,
                    lambda_sparsity=lambda_sparsity,
                    beta=beta,
                    tau=tau
                )
    
                loss = recon_loss + kl_loss + sparsity_loss + lagrangian_term
                loss.backward(retain_graph=True)
                optimizer.step()
    
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print("NaN or Inf detected in loss function!")
                    print(f"Recon loss: {recon_loss}, KL loss: {kl_loss}, Sparsity loss: {sparsity_loss}, Acyclicity loss: {acyclicity_loss}, Attention loss: {attention_loss}")
                    exit(1)
    
                total_loss += loss.item()
                batch_losses.append(loss.item())  # Store batch loss
            
        else:
            for batch_idx, (x_batch) in enumerate(dataloader):
                if x_batch.shape[0]<=x_batch.shape[2]:
                    continue
    
                optimizer.zero_grad()
    
                # Move data to correct device and convert to float32
                x_batch = x_batch.to(device).float()
                #entity_batch = entity_batch.to(device).float()
                #time_batch = time_batch.to(device).float()
    
                # Forward pass
                recon_x, mu, logvar, W, A = model(x_batch, None, None, num_nodes=x_batch.shape[1])

                recon_loss, kl_loss, sparsity_loss, lagrangian_term, h_val = augmented_lagrangian_loss(
                    recon_x.float(),
                    x_batch[:,0,:].float(),
                    mu.float(),
                    logvar.float(),
                    W.float(),
                    model,
                    lambda_sparsity=1e-3,
                    beta=beta,
                    tau=tau
                )
    
                loss = recon_loss + kl_loss + sparsity_loss + lagrangian_term
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
            #model.rho = min(model.rho * 1.5,1e7)
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
            print(f"Recon Loss ={recon_loss}, KL Loss = {kl_loss:.4f}, Sparsity Loss = {sparsity_loss:.4f}, Lagrangian Loss = {lagrangian_term:.4f}")

        # Early stopping condition
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch} epochs. Restoring best model.")
            model.load_state_dict(best_model_state)  # Restore best model state
            break
            

    return loss_history  # Return trained model, epoch losses, and batch losses
