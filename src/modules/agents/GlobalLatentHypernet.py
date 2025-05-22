import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalLatentHyperNet(nn.Module):
    def __init__(self, state_dim, obs_dim, latent_dim, hidden_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        # Fixed first layer of embed_net
        self.embed_layer1 = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU()
        )

        # Encode global state to latent distribution (mean + logvar)
        self.global_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)
        )

        # Decode latent vector to weights for final layer only
        self.weight_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * (2 * latent_dim) + 2 * latent_dim)
        )

    def sample_latent(self, global_state):
        z_param = self.global_encoder(global_state)  # [B, 2 * latent_dim]
        mean, logvar = z_param[:, :self.latent_dim], z_param[:, self.latent_dim:]
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
        return z, kl

    def apply_final_layer(self, h, weights_vector):
        B = h.size(0)
        output_dim = 2 * self.latent_dim
        offset = 0

        W = weights_vector[:, offset:offset + self.hidden_dim * output_dim].view(B, output_dim, self.hidden_dim)
        offset += self.hidden_dim * output_dim
        b = weights_vector[:, offset:offset + output_dim].view(B, output_dim)

        out = torch.bmm(h.unsqueeze(1), W.transpose(1, 2)).squeeze(1) + b
        return out

    def forward(self, global_state):
        """
        global_state: [B, state_dim]
        agent_obs: [B, obs_dim]
        """
        z, kl = self.sample_latent(global_state)
        weights = self.weight_decoder(z)  # [B, final_layer_params]
        # h = self.embed_layer1(agent_obs)  # Fixed layer 1
        # latent = self.apply_final_layer(h, weights)  # Generated layer 2
        return weights, kl
