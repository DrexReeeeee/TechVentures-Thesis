"""
evidence_aware_shrinkage_v2.py
=================================
Experiment 20 (P3-v2): fixed-reference evidence-aware cluster shrinkage.

Why v2: Experiment 19's region_cluster_shrinkage_penalty_evidence_aware
used ONE Optuna-searched scalar cluster_evidence_k shared by all three
clusters (rho_c = n_c/(n_c+k)). The 10-seed multiseed check
(multiseed_13_vs_19_raw*.csv) showed Southeast Asia improving genuinely
and consistently -- but Africa regressed too, and more consistently than
noise alone explains. Likely cause: Optuna's single k has to compromise
across all three clusters at once. Exp19's chosen k=11,392 gave
Europe & Africa rho_c=0.32 and Americas rho_c=0.60 -- damping clusters
that already had a reliable anchor (Europe, North America) as a side
effect of damping Asia-Pacific (rho_c=0.15) enough to help Southeast
Asia. That collateral damping of Europe & Africa is a plausible direct
cause of Africa's regression.

Fix: replace the single tunable k with a FIXED, a-priori reference --
no Optuna search, no knob that could compromise across clusters.
Reference point: Europe's own training-row count (5,342) -- the
smallest cluster-anchor size this dataset gives us a priori reason to
trust (it already carries Africa to be SFTNN's best region in
Experiment 13). A cluster whose total training evidence is >= that
reference is treated as FULLY reliable (rho_c capped at 1.0 -- i.e.
identical to Experiment 13's original flat-strength penalty, no damping
at all). Only a cluster with LESS evidence than that reference gets
damped, proportionally.

    rho_c = min(1.0, n_c / EUROPE_ANCHOR_N)

With this dataset's actual cluster totals (train split, from
05_train_sftnn.py's region_counts):
    Americas        n_c=17,331  -> rho_c = 1.0   (unchanged from Exp13)
    Europe & Africa n_c= 5,448  -> rho_c = 1.0   (unchanged from Exp13 --
                                                   Africa's penalty term
                                                   is IDENTICAL to
                                                   Experiment 13's)
    Asia-Pacific    n_c= 1,987  -> rho_c = 0.372 (damped -- this is the
                                                   only cluster this
                                                   change touches)

This is chosen before looking at any Experiment 20 results (it only uses
Europe's already-known, pre-existing training-row count), so it is not
tuned to flatter Southeast Asia -- and because Americas/Europe & Africa
land at the rho_c=1.0 ceiling, Africa's and North America's shrinkage
behavior under this penalty is mathematically IDENTICAL to Experiment
13's original (unweighted) cluster penalty. Only Asia-Pacific's members
(East Asia, South Asia, Southeast Asia) are affected at all.
"""
import torch

from models import SFTNN

EUROPE_ANCHOR_N = 5342.0


class SFTNNEvidenceAwareClusterFixed(SFTNN):
    """SFTNN with one additional method: a FIXED-reference evidence-aware
    cluster penalty (no tunable k). Forward pass, embedding, affine
    generator, identity-centered init, and region_shrinkage_penalty
    (global-mean pull) are all inherited unchanged from models.SFTNN."""

    def region_cluster_shrinkage_penalty_evidence_aware_fixed(self, region_counts, cluster_ids,
                                                                anchor_n=EUROPE_ANCHOR_N):
        emb = self.region_embedding.weight                    # [n_regions, embed_dim]
        n_clusters = int(cluster_ids.max().item()) + 1
        cluster_means = torch.stack([
            emb[cluster_ids == c].mean(dim=0) for c in range(n_clusters)
        ])                                                     # [n_clusters, embed_dim]
        region_cluster_mean = cluster_means[cluster_ids]        # [n_regions, embed_dim]
        sq_dist = ((emb - region_cluster_mean) ** 2).mean(dim=1)  # [n_regions]

        region_weights = 1.0 / region_counts.float().clamp(min=1.0)
        region_weights = region_weights / region_weights.sum()

        cluster_totals = torch.zeros(n_clusters, device=region_counts.device)
        cluster_totals.scatter_add_(0, cluster_ids, region_counts.float())
        rho_c = (cluster_totals / anchor_n).clamp(max=1.0)      # [n_clusters]
        rho_per_region = rho_c[cluster_ids]                      # [n_regions]

        weights = region_weights * rho_per_region
        return (weights * sq_dist).sum()
