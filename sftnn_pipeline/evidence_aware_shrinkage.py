"""
evidence_aware_shrinkage.py
=============================
Experiment 19 (P3): evidence-aware cluster shrinkage.

Diagnosis this addresses: Experiment 13's region_cluster_shrinkage_penalty
(models.py) pulls every region's embedding toward its CLUSTER mean with
one flat cluster_shrink_lambda, regardless of how reliable that cluster
mean actually is. Using the a-priori REGION_CLUSTERS map (05_train_sftnn.py):

    Americas        = {North America (16,799 train rows), Latin America (532)}
    Europe & Africa  = {Europe (5,342), Africa (106)}
    Asia-Pacific     = {East Asia (1,086), South Asia (566), Southeast Asia (335)}

Africa's cluster mean is anchored by Europe, a large, stable region (5,342
rows) -- a good thing to shrink toward. Southeast Asia's cluster mean has
no such anchor: its cluster-mates (East Asia, South Asia) are themselves
small (max 1,086 rows, 15x smaller than Europe). Pulling Southeast Asia
toward that cluster mean with the SAME strength Africa gets pulled toward
Europe is pooling noise with noise, not borrowing strength from a reliable
anchor -- and this is exactly the asymmetry the Experiment 13 test-set
results showed (Africa: SFTNN's best region; Southeast Asia: SFTNN's
worst region relative to baselines).

Fix, standard empirical-Bayes / James-Stein partial pooling: scale each
region's pull toward its cluster mean by rho_c = n_c / (n_c + k), where
n_c is the CLUSTER's total training evidence (not the region's own n_r,
which the existing 1/n_r per-region weighting already accounts for).
A cluster with little total evidence (Asia-Pacific) gets rho_c near 0, so
its members' cluster-mean pull is automatically damped -- they fall back
on the EXISTING region_shrinkage_penalty's pull toward the global mean
instead (untouched by this change). A cluster anchored by a large region
(Europe & Africa, Americas) gets rho_c near 1, leaving that cluster's
behavior close to Experiment 13's original, unweighted penalty.

k is itself an Optuna-searched hyperparameter (not hand-picked to flatter
any particular region), consistent with how shrink_lambda/
cluster_shrink_lambda are already searched rather than fixed.

This does NOT touch models.py, 05_train_sftnn.py, REGION_CLUSTERS'
membership, or region_shrinkage_penalty -- only adds one new method via
subclassing, used by a separate Experiment 19 training script.
"""
import torch

from models import SFTNN


class SFTNNEvidenceAwareCluster(SFTNN):
    """SFTNN with one additional method: an evidence-aware version of
    region_cluster_shrinkage_penalty. Forward pass, embedding, affine
    generator, identity-centered init, and the original
    region_shrinkage_penalty (global-mean pull) are all inherited
    unchanged from models.SFTNN."""

    def region_cluster_shrinkage_penalty_evidence_aware(self, region_counts, cluster_ids, k):
        emb = self.region_embedding.weight                    # [n_regions, embed_dim]
        n_clusters = int(cluster_ids.max().item()) + 1
        cluster_means = torch.stack([
            emb[cluster_ids == c].mean(dim=0) for c in range(n_clusters)
        ])                                                     # [n_clusters, embed_dim]
        region_cluster_mean = cluster_means[cluster_ids]        # [n_regions, embed_dim]
        sq_dist = ((emb - region_cluster_mean) ** 2).mean(dim=1)  # [n_regions]

        # Same 1/n_r per-region weighting as the original penalty --
        # small regions still matter more than large ones within this term.
        region_weights = 1.0 / region_counts.float().clamp(min=1.0)
        region_weights = region_weights / region_weights.sum()

        # NEW: damp each region's cluster-mean pull by how much total
        # evidence its CLUSTER carries, not its own n_r alone.
        cluster_totals = torch.zeros(n_clusters, device=region_counts.device)
        cluster_totals.scatter_add_(0, cluster_ids, region_counts.float())
        rho_c = cluster_totals / (cluster_totals + k)           # [n_clusters]
        rho_per_region = rho_c[cluster_ids]                      # [n_regions]

        weights = region_weights * rho_per_region
        return (weights * sq_dist).sum()
