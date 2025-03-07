import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GATConv
from torch_geometric.utils import dense_to_sparse
from torch.distributions import Normal, Laplace, RelaxedOneHotCategorical
from torchdiffeq import odeint  # For continuous-time normalizing flows
BATCH_SIZE = 64
TIME_STEPS = 3

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

def notears_constraint(W):
    """NoTears DAG constraint: trace(exp(W * W)) - d"""
    d = W.shape[0]
    expm_ww = torch.matrix_exp(W * W)
    return torch.trace(expm_ww) - d

class FourierTimeEmbedding(nn.Module):
    """Encodes time indices using sinusoidal embeddings for better temporal representation."""
    def __init__(self, embedding_dim,time_steps, max_time=100):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_time = max_time
        
        # Create frequency bands (log-spaced)
        self.freqs = torch.exp(torch.linspace(0, time_steps, embedding_dim//2)) 

    def forward(self, time_indices):
        """
        Args:
            time_indices: Tensor of shape [batch_size, time_steps] containing integer time indices
        Returns:
            Embedded time representations of shape [batch_size, time_steps, embedding_dim]
        """
        time_indices = time_indices.unsqueeze(-1)  # Shape [batch_size, time_steps, 1]
        sinusoidal_in = time_indices * self.freqs.to(time_indices.device)  # Shape [batch_size, time_steps, embedding_dim//2]
        time_embedding = torch.cat([torch.sin(sinusoidal_in), torch.cos(sinusoidal_in)], dim=-1)
        return time_embedding  # Shape [batch_size, time_steps, embedding_dim]

class TemporalRealNVPFlow(nn.Module):
    """Time-Adaptive Normalizing Flow for Latent Confounders."""
    def __init__(self, latent_dim,input_dim):
        super().__init__()
        self.scale = nn.Linear(latent_dim // 2, latent_dim // 2)
        self.translate = nn.Linear(latent_dim // 2, latent_dim // 2)
        self.time_embedding = FourierTimeEmbedding(latent_dim,input_dim)
        self.temporal_gate = nn.GRU(latent_dim, latent_dim//2, batch_first=True)  # Time-aware updates
        self.time_projection = nn.Linear(input_dim,1)
    def forward(self, z, time_context):
        z1, z2 = z.chunk(2, dim=1)  # Split into two parts
        s = torch.sigmoid(self.scale(z1))
        t = self.translate(z1)
        z2 = s * z2 + t
        time_embed = self.time_embedding(time_context)
        # Temporal adjustment to confounders
        time_out, _ = self.temporal_gate(time_embed)
        time_out = self.time_projection(time_out.permute(0,2,1)).squeeze(2)
        z2 = z2 + time_out#.squeeze(1)  # Adjust for temporal shift
        
        return torch.cat([z1, z2], dim=1)

class TemporalCausalGraph(nn.Module):
    """
    Implements a Temporal Causal Graph (TCG) with:
    - Time-dependent adjacency matrix (instantaneous + delayed effects)
    - Adaptive normalizing flow for non-stationarity handling
    """
    def __init__(self, num_nodes, hidden_dim, latent_dim,time_steps=10, prior_adj=None, mixed_data=False):
        super(TemporalCausalGraph, self).__init__()
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        self.time_steps = time_steps
        self.mixed_data = mixed_data  # Support for categorical + continuous

        # Learnable adjacency matrices (instantaneous + delayed)
        self.edge_score_now = nn.Parameter(torch.randn(num_nodes, num_nodes))
        self.edge_score_lag = nn.Parameter(torch.randn(num_nodes, num_nodes))
        self.prior_adj = prior_adj if prior_adj is not None else torch.zeros(num_nodes, num_nodes)

        # Direct adjacency learning
        self.x_projection1 = nn.Linear(num_nodes,hidden_dim)
        self.self_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model = hidden_dim,nhead=4,dim_feedforward=hidden_dim),num_layers=2)
        self.x_projection2 = nn.Linear(hidden_dim,num_nodes)

        # Latent Confounders Z (Time-Aware)
        
        self.temporal_flow = TemporalRealNVPFlow(latent_dim,time_steps)

        # Temporal Graph Attention Network
        self.gnn = GATv2Conv(num_nodes, hidden_dim, heads=4, concat=True)

        # Mapping from latent space to num_nodes
        self.latent_to_nodes = nn.Linear(latent_dim, num_nodes)

        # Likelihood Models
        self.gaussian_likelihood = nn.Linear(hidden_dim * 4, self.num_nodes*2)  # Mean, Log-Variance
        if mixed_data:
            self.categorical_likelihood = nn.Linear(hidden_dim, num_nodes)  # Gumbel-Softmax Output

    def forward(self, X, time_context,Z):
        """ Learns causal graph over time and performs inference """
        # Compute adjacency matrices
        x = self.x_projection1(X)
        X_transformed = self.self_attention(x.permute(1,0,2))
        x = self.x_projection2(X_transformed)
        x_instantaneous = x[0,:,:]
        x_lag = x[1:,:,:].mean(dim=0)

        adj_learned_now = F.gumbel_softmax(torch.einsum('bk,bj->kj', x_instantaneous, x_instantaneous), tau=1, hard=False)
        adj_learned_lag = F.gumbel_softmax(torch.sigmoid(torch.einsum('bk,bj->kj', x_lag, x_lag)), tau=1, hard=False)
                    
        adj_now = torch.sigmoid(self.edge_score_now + self.prior_adj + adj_learned_now)
        adj_lag = torch.sigmoid(self.edge_score_lag + self.prior_adj + adj_learned_lag)
        adj_now = torch.clamp(adj_now, 0, 1)
        adj_lag = torch.clamp(adj_lag, 0, 1)
        adj_now = replace_zero(adj_now.fill_diagonal_(0))  # No self-loops
        adj_lag = replace_zero(adj_lag.fill_diagonal_(0))  # No self-loops in lag structure
        
        # Encode latent confounders with time-awareness
        #Z = self.latent_Z + torch.randn_like(self.latent_Z) * 0.1
        Z = self.temporal_flow(Z, time_context)  # Apply time-adaptive normalizing flow
        Z = self.latent_to_nodes(Z)  # Map latent space to num_nodes

        # Temporal graph attention
        edge_index, _ = dense_to_sparse(adj_now + adj_lag)
        X_emb = self.gnn(Z, edge_index)
        X_emb = X_emb.view(X_emb.shape[0], -1)

        # Likelihood computation
        mean_logvar = self.gaussian_likelihood(X_emb)
        mean, log_var = torch.split(mean_logvar, mean_logvar.shape[-1] // 2, dim=-1)
        log_var = torch.clamp(log_var, -5, 2)  # Stabilization
        
        likelihood = Laplace(mean, torch.exp(0.5 * log_var))
        
        return adj_now, adj_lag, likelihood


class CausalGraphVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_nodes, time_steps=10, prior_adj=None):
        super(CausalGraphVAE, self).__init__()
        self.time_steps = time_steps
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        self.causal_graph = TemporalCausalGraph(num_nodes, hidden_dim, latent_dim, time_steps, prior_adj)
        self.alpha = torch.tensor(0.0, dtype=torch.float32, requires_grad=False)
        self.rho = torch.tensor(1.0, dtype=torch.float32, requires_grad=False)

        # Temporal-aware Encoder and Decoder
        self.encoder_rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        self.decoder_rnn = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.decoder_fc = nn.Linear(hidden_dim, input_dim)

    def encode(self, X, time_context):

        X_enc, _ = self.encoder_rnn(X)
        mu, logvar = self.mu_layer(X_enc[:, -1, :]), self.logvar_layer(X_enc[:, -1, :])
        Z = mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)
        adj_now, adj_lag, likelihood = self.causal_graph(X, time_context,Z)  # Use temporal causal graph
        return mu, logvar, adj_now, adj_lag, likelihood

    def decode(self, Z, adj_now, adj_lag):
        Z_expanded = Z.unsqueeze(1).repeat(1, self.time_steps, 1)
        X_dec, _ = self.decoder_rnn(Z_expanded)
        X_dec = self.decoder_fc(X_dec)
        return X_dec

    def forward(self, X, time_context):
        mu, logvar, adj_now, adj_lag, likelihood = self.encode(X, time_context)
        Z = mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)
        recon_X = self.decode(Z, adj_now, adj_lag)
        return recon_X, mu, logvar, adj_now, adj_lag, likelihood

    def infer_causal_effect(self,X_data,T_data, target_variable,labels):
        """Infers the top 3 causal factors for a given target variable based on counterfactual analysis."""

        try:
            target_idx = col_names.index(target_variable, 0)
        except:
            return target_variable + " not in index"
        self.eval()
        n_samples = X_data.shape[0]

        with torch.no_grad():
            _, _, adj_now, adj_lag, _ = self.encode(X_data,T_data)
            adj_matrix = adj_now + adj_lag  # Aggregate instantaneous and delayed effects
            adj_matrix = torch.sigmoid(adj_matrix)
            adj_matrix=adj_matrix.fill_diagonal_(0)

            l, v = np.linalg.eig(adj_matrix.cpu().detach().numpy().T)
            score = (v[:,0]/np.sum(v[:,0])).real
            score=np.around(score/np.sum(score), decimals=3)
            root_rank = pd.DataFrame(score,index=col_names,columns=['score']).sort_values(by='score',ascending=False)

            
            causal_strengths = adj_matrix[:, target_idx].cpu().detach().numpy()
            edge_strengths = {}
            for i, val in enumerate(causal_strengths):
                edge_strengths[labels[i]] = float(np.round(val.item(),6))
            
            counterfactual_results = {}
            for i in range(self.num_nodes):
                if i != target_idx:
                    label = labels[i]
                    intervention = torch.zeros((n_samples,self.time_steps ,self.num_nodes))
                    intervention[:,:, i] = 1  # Set do-intervention on cause variable
                    mu, logvar, adj_now_int, adj_lag_int, likelihood = self.encode(intervention, T_data)
                    Z = mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)
                    recon_X = self.decode(Z, adj_now_int, adj_lag_int)
            
                    counter_factual_adj = adj_now_int+ adj_lag_int
                    counterfactual_results[label] = torch.norm(X_data[:, :, target_idx] - recon_X[:, :, target_idx]).item()
                        
            sorted_keys = sorted(counterfactual_results, key=counterfactual_results.get, reverse=True)
            counterfactual_rankings = {key: counterfactual_results[key] for key in sorted_keys}
            sorted_keys = sorted(edge_strengths, key=edge_strengths.get, reverse=True)
            top_causes = {key: edge_strengths[key] for key in sorted_keys}
        return top_causes, counterfactual_rankings, root_rank

    def loss_function(self, recon_X, X, mu, logvar, likelihood, adj_now, adj_lag, epoch, max_epochs,rho_max=30.0,alpha_max=15.0):
        """Loss function including reconstruction, KL divergence, DAG penalty, and sparsity with annealing."""
        recon_loss = F.mse_loss(recon_X, X, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        sparsity_loss = torch.norm(adj_now, p=1) + torch.norm(adj_lag, p=1)
        
        # DAG constraint (NoTears) with annealing
        h_value = notears_constraint(adj_now) + notears_constraint(adj_lag)
        
        # Annealing schedule for rho and alpha
        rho_max = 30.0  # Maximum value for rho
        rho_min = 1.0   # Minimum value for rho
        alpha_max = 15.0 # Maximum value for alpha
        alpha_min = 0.0 # Minimum value for alpha

        likelihood_loss = (-likelihood.log_prob(X[:,0,:]).sum())/X.numel()
        
        self.rho = rho_min + (rho_max - rho_min) * (epoch / max_epochs)
        self.alpha = alpha_min + (alpha_max - alpha_min) * (epoch / max_epochs)
        
        lagrangian_loss = (self.alpha * h_value + 0.5 * self.rho * (h_value ** 2))/X.numel()
        

        return recon_loss , kl_loss , sparsity_loss , lagrangian_loss, likelihood_loss

    def train_model(self, dataloader, optimizer, num_epochs=100, patience=10,BATCH_SIZE=64,rho_max=30.0,alpha_max=15.0):
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, (X_batch, time_batch) in enumerate(dataloader):
                if X_batch.shape[0] != BATCH_SIZE:
                    padding_needed = max(0, BATCH_SIZE - time_batch.size(0))
                    time_padding = (0, 0,padding_needed,0)  # (left, right, top, bottom, front, back) for last 3 dims
                    time_batch = F.pad(time_batch, time_padding, mode='constant', value=0)
                    x_padding = (0, 0,0,0,padding_needed,0)  # (left, right, top, bottom, front, back) for last 3 dims
                    X_batch = F.pad(X_batch, x_padding, mode='constant', value=0)
            
                optimizer.zero_grad()
                recon_X, mu, logvar, adj_now, adj_lag, likelihood = self.forward(X_batch, time_batch)
                recon_loss , kl_loss , sparsity_loss , lagrangian_loss, likelihood_loss = self.loss_function(recon_X, X_batch, mu, logvar,likelihood, adj_now, 
                                                                                            adj_lag, epoch, num_epochs,rho_max,alpha_max)
                loss = recon_loss + kl_loss + sparsity_loss + lagrangian_loss  + likelihood_loss
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
                print(f"Recon Loss ={recon_loss}, KL Loss = {kl_loss:.4f}, Sparsity Loss = {sparsity_loss:.4f}, Lagrangian Loss = {lagrangian_loss:.4f}, Likelihood Loss = {likelihood_loss:.4f}") 

            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered. Last Epoch: " + str(epoch) )
                    break




from torch.utils.data import Dataset, DataLoader
import torch

class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe, time_steps=10):
        """
        Args:
            dataframe (pd.DataFrame): Time-series data
            time_steps (int): Number of past time steps to consider
        """
        self.data = torch.tensor(dataframe.values, dtype=torch.float32)
        self.time_steps = time_steps
    
    def __len__(self):
        return len(self.data) - self.time_steps

    def __getitem__(self, idx):
        """
        Returns:
            X: Past `time_steps` values (shape: [time_steps, num_features])
            y: Next step prediction target (shape: [num_features])
        """
        X = self.data[idx : idx + self.time_steps, :]
        time_context = torch.arange(self.time_steps).float()  # Time indexing
        
        return X, time_context

try:
    bad = bad.drop('time',axis=1)
except:
    None
dataset = TimeSeriesDataset(bad, time_steps=TIME_STEPS)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
X_data = torch.empty(0)
T_data = torch.empty(0)
for batch_idx, (X_batch, time_batch) in enumerate(dataloader):
    X_data = torch.cat((X_data[:batch_idx], X_batch, X_data[batch_idx:]))
    T_data = torch.cat((T_data[:batch_idx], time_batch, T_data[batch_idx:]))

model = CausalGraphVAE(input_dim=bad.shape[1], hidden_dim=256, latent_dim=16, num_nodes=bad.shape[1],time_steps=TIME_STEPS)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train_model(dataloader, optimizer, num_epochs=15000, patience=100,BATCH_SIZE=BATCH_SIZE,rho_max=30.0,alpha_max=15.0)
