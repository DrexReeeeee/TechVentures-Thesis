"""
deep_film_sftnn.py
=====================
Experiment 22: Deep FiLM -- region conditioning at EVERY trunk layer,
not just the last one.

Diagnosis: the original SFTNN (models.py) modulates the trunk's output
ONCE, right before the head: h = trunk(x); y = gamma*h + beta. Every
earlier layer's feature extraction is completely region-blind -- forced
to find ONE shared nonlinear transformation that works for all 7 regions
simultaneously, with region-awareness entering only as a final linear
rescale+shift. XGBoost, given region as a feature, can let it interact
with any other feature at any depth of any tree, in deeply nonlinear,
conditional ways -- a structurally richer form of region-conditioning.
Every mechanism tested so far (Experiments 17-21) only ever tuned the
STRENGTH of SFTNN's single existing modulation point; none changed how
many points exist. This may be why SFTNN has never won the pooled
aggregate against XGBoost regardless of how that one point is
regularized.

Fix, following the ORIGINAL FiLM paper's actual design (Perez et al.,
2017, "FiLM: Visual Reasoning with a General Conditioning Layer") --
which conditions every conv block, not just the last: apply an
identity-centered affine modulation after EVERY trunk layer. All layers
share ONE region embedding (so the existing, already-validated
region_shrinkage_penalty needs no change -- same embedding, same global-
mean pull); each layer gets its own small linear readout from that
embedding to its own gamma/beta, zero-initialized so every layer starts
as an exact no-op for every region (gamma=1, beta=0), exactly matching
the original SFTNN's identity-centered init principle.

Kept deliberately simple for the first test of this idea: each layer's
readout is a single Linear(embed_dim, 2*hidden_dim) (not a 2-layer MLP
like the original affine_generator) to limit how much new capacity is
added relative to the ~36k-parameter trunk, and only the existing
global-mean shrinkage penalty is used (no cluster-level shrinkage
layered on top yet) -- isolating whether deepening the conditioning
itself helps before combining it with any of today's other validated
mechanisms (calibration, per-cluster lambda).
"""
import torch
import torch.nn as nn


class DeepFiLMSFTNN(nn.Module):
    def __init__(self, n_features, n_regions, hidden_dim=64, n_layers=2,
                 embed_dim=8, dropout=0.2):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i in range(n_layers):
            in_dim = n_features if i == 0 else hidden_dim
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))

        self.region_embedding = nn.Embedding(n_regions, embed_dim)

        # One small, linear, identity-centered FiLM readout PER LAYER,
        # all reading from the single shared region embedding.
        self.film_readouts = nn.ModuleList()
        for i in range(n_layers):
            readout = nn.Linear(embed_dim, 2 * hidden_dim)
            nn.init.zeros_(readout.weight)
            nn.init.zeros_(readout.bias)
            self.film_readouts.append(readout)

        self.gamma_scale = nn.Parameter(torch.tensor(1.0))
        self.beta_scale = nn.Parameter(torch.tensor(1.0))

        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, region_id, return_affine=False):
        e_r = self.region_embedding(region_id)
        h = x
        all_gamma, all_beta = [], []
        for i in range(self.n_layers):
            h = torch.relu(self.layers[i](h))
            gamma_beta = self.film_readouts[i](e_r)
            gamma_raw, beta_raw = gamma_beta.chunk(2, dim=-1)
            gamma = 1.0 + self.gamma_scale * torch.tanh(gamma_raw)
            beta = self.beta_scale * torch.tanh(beta_raw)
            h = gamma * h + beta
            h = self.dropouts[i](h)
            if return_affine:
                all_gamma.append(gamma)
                all_beta.append(beta)
        logits = self.head(h).squeeze(-1)
        if return_affine:
            return logits, all_gamma, all_beta
        return logits

    def region_shrinkage_penalty(self, region_counts):
        """Unchanged from models.SFTNN -- same single shared embedding,
        same already-validated global-mean shrinkage penalty."""
        emb = self.region_embedding.weight
        mean_emb = emb.mean(dim=0, keepdim=True)
        sq_dist = ((emb - mean_emb) ** 2).mean(dim=1)
        weights = 1.0 / region_counts.float().clamp(min=1.0)
        weights = weights / weights.sum()
        return (weights * sq_dist).sum()
