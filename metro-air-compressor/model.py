import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    """
    LSTM-based autoencoder for temporal pattern learning.
    """
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.latent = nn.Linear(hidden_dim, latent_dim)
        self.decoder_lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: (batch, seq_len, features)
        encoded_seq, _ = self.encoder(x)
        latent = self.latent(encoded_seq[:, -1, :]).unsqueeze(1).repeat(1, x.size(1), 1)
        decoded_seq, _ = self.decoder_lstm(latent)
        out = self.output_layer(decoded_seq)
        return out
