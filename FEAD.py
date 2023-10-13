import torch
import torch.nn as nn
import torch.nn.functional as F

class FEAD(nn.Module):
    def __init__(self, input_dim, cnn_out_dim, hidden_dim, latent_dim, condition_dim):
        super(FEAD, self).__init__()

        # Encoder
        self.conv1 = nn.Conv1d(input_dim, cnn_out_dim, kernel_size=3, stride=1, padding=1)
        self.lstm1 = nn.LSTM(cnn_out_dim + condition_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.lstm2 = nn.LSTM(latent_dim + condition_dim, hidden_dim, batch_first=True)
        self.deconv1 = nn.ConvTranspose1d(hidden_dim, input_dim, kernel_size=3, stride=1, padding=1)

    def encode(self, x, condition):
        # 1D-CNN
        x = F.relu(self.conv1(x))

        # Combine CNN output and condition
        repeated_condition = condition.unsqueeze(1).repeat(1, x.size(1), 1)
        x = torch.cat([x, repeated_condition], dim=-1)

        # LSTM
        _, (h_n, _) = self.lstm1(x)

        # Linear layers
        mu = self.fc_mu(h_n.squeeze(0))
        log_var = self.fc_var(h_n.squeeze(0))

        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, condition):
        # Combine latent variable and condition
        repeated_condition = condition.unsqueeze(1).repeat(1, z.size(1), 1)
        z = torch.cat([z, repeated_condition], dim=-1)

        # LSTM
        x, _ = self.lstm2(z)

        # 1D-CNN Transpose
        x = self.deconv1(x)

        return torch.sigmoid(x)

    def forward(self, x, condition):
        mu, log_var = self.encode(x, condition)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z.unsqueeze(1), condition)
        return x_recon, mu, log_var

