"""
models.py
=========
Shared PyTorch model definitions used by 02_train_mlp.py and
05_train_sftnn.py.

- StandardMLP  : region-blind base control model (Section 4.7, "Standard
                 MLP (Region-Blind Base Control)"). Region is one-hot
                 encoded and concatenated into the input layer.
- SFTNN        : the proposed Spatial Feature Transformation Neural
                 Network (Section 4.5.2). Region is NOT concatenated into
                 x; instead it drives a separate embedding + affine
                 parameter generator that modulates the shared trunk's
                 hidden representation (Equations 1-4).
"""
import torch
import torch.nn as nn


class StandardMLP(nn.Module):
    """Region-blind base control MLP (Section 4.7).
    Input = [x ; one_hot(region)] -> hidden layers -> sigmoid."""

    def __init__(self, n_features, n_regions, hidden_dim=64, n_layers=2, dropout=0.2):
        super().__init__()
        in_dim = n_features + n_regions
        layers = []
        for i in range(n_layers):
            layers += [nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                       nn.ReLU(), nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, 1)
        self.n_regions = n_regions

    def forward(self, x, region_id):
        region_onehot = nn.functional.one_hot(region_id, num_classes=self.n_regions).float()
        h = self.trunk(torch.cat([x, region_onehot], dim=1))
        return self.head(h).squeeze(-1)  # logits


class SFTNN(nn.Module):
    """
    Proposed Spatial Feature Transformation Neural Network (Section 4.5.2).

    Equation 1 (Shared Trunk Forward Pass):
        h^(l) = ReLU(W^(l) h^(l-1) + b^(l)),  h^(0) = x
    Equation 2 (Regional Scale and Shift Generation):
        [gamma_r; beta_r] = W . ReLU(W_e e_r + b_e) + b
    Equation 3 (SFTNN Affine Transformation):
        y_modulated = gamma_r (*) h + beta_r
    Equation 4 (Output Probability and Loss):
        p = sigmoid(w_o^T y_modulated + b_o)
    """

    def __init__(self, n_features, n_regions, hidden_dim=64, n_layers=2,
                 embed_dim=8, dropout=0.2):
        super().__init__()
        # --- Shared Startup Feature Trunk (Equation 1) ---
        layers = []
        for i in range(n_layers):
            layers += [nn.Linear(n_features if i == 0 else hidden_dim, hidden_dim),
                       nn.ReLU(), nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*layers)

        # --- Region Embedding + Affine Parameter Generator (Equation 2) ---
        self.region_embedding = nn.Embedding(n_regions, embed_dim)
        self.affine_generator = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),  # outputs [gamma ; beta]
        )
        self.hidden_dim = hidden_dim

        # --- Classifier Head (Equation 4) ---
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, region_id, return_affine=False):
        h = self.trunk(x)                                  # Equation 1
        e_r = self.region_embedding(region_id)              # lookup e_r = E[r]
        gamma_beta = self.affine_generator(e_r)              # Equation 2
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        y_modulated = gamma * h + beta                       # Equation 3
        logits = self.head(y_modulated).squeeze(-1)          # Equation 4
        if return_affine:
            return logits, gamma, beta
        return logits

    def get_region_gamma_beta(self, device="cpu"):
        """Extract gamma_r/beta_r for every region id (Section 4.10.2
        interpretability heatmaps) without needing any startup features."""
        with torch.no_grad():
            all_ids = torch.arange(self.region_embedding.num_embeddings, device=device)
            e_r = self.region_embedding(all_ids)
            gamma_beta = self.affine_generator(e_r)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma.cpu().numpy(), beta.cpu().numpy()
