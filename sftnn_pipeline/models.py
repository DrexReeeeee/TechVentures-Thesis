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
            nn.Linear(hidden_dim, 2 * hidden_dim),  # outputs [gamma_raw ; beta_raw]
        )
        # --- Identity-centered FiLM init (fix #1) ---
        # Zero-init the final affine_generator layer so gamma_raw = beta_raw = 0
        # for every region at the start of training. Combined with the
        # reparameterization in forward() below (gamma = 1 + tanh(gamma_raw),
        # beta = tanh(beta_raw)), this means every region starts as an exact
        # no-op on the trunk (gamma=1, beta=0) instead of the old raw-gamma
        # parameterization, which started near gamma=0 -- i.e. near-erasing
        # the trunk's representation before any modulation was learned. That
        # previously disadvantaged minority regions most, since they have the
        # least gradient signal available to climb back out of a collapsed
        # state. Now the network only adds region-specific modulation where
        # the gradient actually earns it.
        nn.init.zeros_(self.affine_generator[-1].weight)
        nn.init.zeros_(self.affine_generator[-1].bias)

        # Learnable global scale so the model can still express large
        # modulation if the data calls for it, while every region still
        # starts bounded around the identity transform.
        self.gamma_scale = nn.Parameter(torch.tensor(1.0))
        self.beta_scale = nn.Parameter(torch.tensor(1.0))

        self.hidden_dim = hidden_dim

        # --- Classifier Head (Equation 4) ---
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, region_id, return_affine=False):
        h = self.trunk(x)                                  # Equation 1
        e_r = self.region_embedding(region_id)              # lookup e_r = E[r]
        gamma_beta = self.affine_generator(e_r)              # Equation 2
        gamma_raw, beta_raw = gamma_beta.chunk(2, dim=-1)
        # Identity-centered reparameterization (fix #1): gamma starts at 1
        # (no-op scale) and beta starts at 0 (no-op shift) for every region,
        # bounded via tanh so training stays stable, but scaled by a
        # learnable factor so the model can still express strong modulation.
        gamma = 1.0 + self.gamma_scale * torch.tanh(gamma_raw)
        beta = self.beta_scale * torch.tanh(beta_raw)
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
            gamma_raw, beta_raw = gamma_beta.chunk(2, dim=-1)
            gamma = 1.0 + self.gamma_scale * torch.tanh(gamma_raw)
            beta = self.beta_scale * torch.tanh(beta_raw)
        return gamma.cpu().numpy(), beta.cpu().numpy()

    def region_shrinkage_penalty(self, region_counts):
        """
        Fix #3: partial pooling / shrinkage on region embeddings.

        Every region currently gets a fully free embedding vector, with no
        relationship enforced between regions -- so a small-n region (e.g.
        Southeast Asia, South Asia) can move just as far from the pack as a
        large-n region (North America), even though it doesn't have enough
        data to reliably tell a *good* move from a *bad* one. The
        counterfactual ablation confirmed this concretely: swapping in the
        generic mean embedding improved Recall for Southeast Asia, East
        Asia, and South Asia, and left Africa unchanged -- only North
        America and Latin America were genuinely better off with their own
        specific embedding.

        This adds a penalty term (added to the training loss, NOT applied
        inside forward()) that pulls each region's embedding toward the
        population-mean embedding, weighted by 1/n_r -- so regions with
        less training data are pulled harder toward "behave like the
        average region," while regions with abundant data (which have
        earned the right to look different, per the ablation) are pulled
        only lightly.

        region_counts: LongTensor/FloatTensor of shape [n_regions], the
        number of TRAINING rows for each region id (0-indexed, matching
        region_embedding's row order).
        """
        emb = self.region_embedding.weight              # [n_regions, embed_dim]
        mean_emb = emb.mean(dim=0, keepdim=True)          # unweighted population mean
        sq_dist = ((emb - mean_emb) ** 2).sum(dim=1)       # [n_regions]

        weights = 1.0 / region_counts.float().clamp(min=1.0)
        weights = weights / weights.sum()                 # normalize: penalty scale
                                                            # doesn't grow/shrink with
                                                            # n_regions or dataset size
        return (weights * sq_dist).sum()