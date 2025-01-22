class LSTMAutoEncoder(nn.Module):
    def __init__(self, n_layers, hidden_size, nb_feature, latent_dim=1, device=torch.device('cpu')):
        """
        Deep LSTM Autoencoder for anomaly detection.
        
        Args:
            n_layers (int): Number of LSTM layers in both encoder and decoder.
            hidden_size (int): Number of hidden units in each LSTM layer.
            nb_feature (int): Number of input features.
            latent_dim (int): Size of the latent space (default=1 for single-dimensional).
            device (torch.device): Device to run the model on.
        """
        super(LSTMAutoEncoder, self).__init__()
        self.device = device
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.nb_feature = nb_feature
        self.latent_dim = latent_dim

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=nb_feature,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False
        )

        # Latent space projection
        self.encoder_fc = nn.Linear(hidden_size, latent_dim)

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False
        )

        # Output projection back to original feature space
        self.decoder_fc = nn.Linear(hidden_size, nb_feature)

    def forward(self, x):
        """
        Forward pass through the LSTM Autoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_length, nb_feature].
        
        Returns:
            torch.Tensor: Reconstructed input of shape [batch_size, seq_length, nb_feature].
        """
        # Encode
        encoded_output, (hidden, cell) = self.encoder_lstm(x)  # hidden shape: [n_layers, batch_size, hidden_size]
        latent_space = self.encoder_fc(hidden[-1])  # Take the last layer's hidden state and project to latent space

        # Expand latent space to match sequence length for the decoder
        latent_sequence = latent_space.unsqueeze(1).repeat(1, x.size(1), 1)  # [batch_size, seq_length, latent_dim]

        # Decode
        decoded_output, _ = self.decoder_lstm(latent_sequence)  # Decoded LSTM output
        reconstructed = self.decoder_fc(decoded_output)  # Map back to feature space

        return reconstructed


from torchinfo import summary

# Example model
model = LSTMAutoEncoder(n_layers=3, hidden_size=128, nb_feature=16, latent_dim=1, device=torch.device('cpu'))

# Print model summary
summary(model, input_size=(32, 50, 16))  # [batch_size, seq_length, nb_feature]


self.model = LSTMAutoEncoder(
    n_layers=self.n_layers,
    hidden_size=self.hidden_size,
    nb_feature=self.nb_feature,
    latent_dim=1,  # Single-dimensional latent space
    device=self.device
)


# Define model parameters
cell_mat = np.random.rand(1000, 50, 16)  # Example input data: [seq_length, size_window, nb_feature]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize autoencoder
autoencoder = lstm_autoencoder(
    cell_mat=cell_mat,
    n_epochs=50,
    n_layers=3,
    hidden_size=128,
    lr=0.001,
    batch_size=32,
    device=device
)

# Train the model
autoencoder.fit()
