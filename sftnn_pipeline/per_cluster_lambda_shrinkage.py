"""
per_cluster_lambda_shrinkage.py
==================================
Experiment 21: independent per-cluster shrinkage strength.

Diagnosis this addresses: Experiment 20 (evidence_aware_shrinkage_v2.py)
made Africa's and Latin America's cluster-shrinkage FORMULA mathematically
identical to Experiment 13's (rho_c=1.0, no damping) -- but Africa still
only partially recovered (F1 .528 Exp13 -> .463 Exp20, across 20 seeds),
because Optuna's single, SHARED cluster_shrink_lambda landed on 39.1 this
run vs Experiment 13's 1.07: a ~36x stronger OVERALL pull toward cluster
means for every cluster, Africa/Latin America included, as a side effect
of whatever made that strength look good for Asia-Pacific. The per-cluster
rho_c controls each cluster's damping RELATIVE to that shared knob; it
cannot insulate a cluster from the knob's ABSOLUTE strength, because it's
still one number all three clusters share.

Fix: give each cluster its own independently Optuna-searched lambda
(lambda_americas, lambda_asia_pacific, lambda_europe_africa) instead of
one shared cluster_shrink_lambda times a fixed rho_c multiplier. This
also makes Experiment 20's rho_c formula unnecessary: with an
independent lambda per cluster, Optuna can directly learn "pull
Asia-Pacific hard, pull Europe & Africa gently" from validation data,
without a hand-crafted anchor constant standing in for that search. So
this reverts to the ORIGINAL, unweighted per-region sq_dist calculation
(same as models.SFTNN.region_cluster_shrinkage_penalty) -- the only
change is that each cluster's contribution is scaled by its OWN lambda,
not a single shared one.
"""
import torch

from models import SFTNN


class SFTNNPerClusterLambda(SFTNN):
    """SFTNN with one additional method: cluster shrinkage with an
    independent strength per cluster. Forward pass, embedding, affine
    generator, identity-centered init, and region_shrinkage_penalty
    (global-mean pull) are all inherited unchanged from models.SFTNN."""

    def region_cluster_shrinkage_penalty_per_cluster(self, region_counts, cluster_ids, cluster_lambdas):
        """
        cluster_lambdas: FloatTensor [n_clusters], one independent
        strength per cluster (cluster id order, matching cluster_ids'
        values -- e.g. [lambda_americas, lambda_asia_pacific,
        lambda_europe_africa] for this dataset's sorted cluster names).
        Returns the ALREADY lambda-weighted penalty -- call this directly
        in the loss, with no separate external multiplier (unlike
        region_shrinkage_penalty / region_cluster_shrinkage_penalty,
        which are multiplied by an external shrink_lambda/
        cluster_shrink_lambda at the call site).
        """
        emb = self.region_embedding.weight                    # [n_regions, embed_dim]
        n_clusters = int(cluster_ids.max().item()) + 1
        cluster_means = torch.stack([
            emb[cluster_ids == c].mean(dim=0) for c in range(n_clusters)
        ])                                                     # [n_clusters, embed_dim]
        region_cluster_mean = cluster_means[cluster_ids]        # [n_regions, embed_dim]
        sq_dist = ((emb - region_cluster_mean) ** 2).mean(dim=1)  # [n_regions]

        region_weights = 1.0 / region_counts.float().clamp(min=1.0)
        region_weights = region_weights / region_weights.sum()

        lambda_per_region = cluster_lambdas[cluster_ids]         # [n_regions]
        weights = region_weights * lambda_per_region
        return (weights * sq_dist).sum()
