"""
ESDE Phase 9: Chaining Metrics
================================

Diagnostic metrics for W5 clustering results.
Quantifies chaining behavior without judging it.

When single-linkage clustering produces a giant island (e.g., 475/492
nodes in one component), these metrics explain WHY:
  - Was it dense connectivity? (high edge_sparsity, high mean_intra_sim)
  - Or chain propagation? (low edge_sparsity, low mean_intra_sim)

"Describe, but do not decide" — metrics explain HOW islands formed,
not whether they should exist.

Spec: Phase 9 Chaining Metrics v1.0
"""

from typing import List, Dict, Any, Tuple, Optional, Set


CHAINING_METRICS_VERSION = "v1.0"


def compute_chaining_metrics(
    islands: List[Dict[str, Any]],
    noise_ids: List[str],
    input_count: int,
    all_sim_pairs: List[Tuple[str, str, float]],
    threshold: float,
) -> Dict[str, Any]:
    """
    Compute chaining diagnostics for W5 clustering output.
    
    Args:
        islands: List of island dicts with 'member_ids' and 'size' keys.
        noise_ids: List of unclustered node IDs.
        input_count: Total number of input nodes.
        all_sim_pairs: ALL pairwise similarities [(a, b, sim), ...].
        threshold: The threshold used for clustering.
        
    Returns:
        Dict with chaining metrics and version.
    """
    metrics: Dict[str, Any] = {
        "version": CHAINING_METRICS_VERSION,
    }
    
    n_total = input_count
    
    if not islands:
        metrics.update({
            "giant_component_ratio": 0.0,
            "largest_island_size": 0,
            "edge_sparsity": 0.0,
            "mean_intra_similarity": None,
            "chaining_detected": False,
        })
        return metrics
    
    # ── Giant Component Ratio ─────────────────────────────
    # Fraction of all nodes in the largest island.
    # Close to 1.0 → nearly everything merged into one blob.
    largest_island = max(islands, key=lambda x: x["size"])
    largest_size = largest_island["size"]
    giant_ratio = round(largest_size / n_total, 4) if n_total > 0 else 0.0
    
    metrics["giant_component_ratio"] = giant_ratio
    metrics["largest_island_size"] = largest_size
    
    # ── Edge Sparsity ─────────────────────────────────────
    # Within the largest island:
    #   actual edges (above threshold) / maximum possible edges
    #
    # High sparsity (close to 1.0) = dense, genuinely similar cluster.
    # Low sparsity (close to 0.0) = sparse, chain-linked cluster.
    member_set: Set[str] = set(largest_island["member_ids"])
    n_members = len(member_set)
    max_edges = n_members * (n_members - 1) // 2
    
    actual_edges = 0
    intra_sims: List[float] = []
    
    for a, b, sim in all_sim_pairs:
        if a in member_set and b in member_set:
            intra_sims.append(sim)
            if sim >= threshold:
                actual_edges += 1
    
    edge_sparsity = round(actual_edges / max_edges, 4) if max_edges > 0 else 0.0
    metrics["edge_sparsity"] = edge_sparsity
    
    # ── Mean Intra-Island Similarity ──────────────────────
    # Average similarity across ALL pairs within the giant island
    # (not just above-threshold edges).
    #
    # If this is much lower than the threshold, nodes are connected
    # through intermediaries, not direct affinity.
    if intra_sims:
        mean_intra = round(sum(intra_sims) / len(intra_sims), 4)
    else:
        mean_intra = None
    
    metrics["mean_intra_similarity"] = mean_intra
    
    # ── Chaining Detection ────────────────────────────────
    # Heuristic: chaining is likely when:
    #   1. Giant component holds >50% of nodes
    #   2. Edge sparsity is low (<0.3)
    #   3. Mean intra-similarity is well below threshold
    #
    # This is a descriptive flag, not a decision.
    chaining_detected = (
        giant_ratio > 0.5
        and edge_sparsity < 0.3
        and mean_intra is not None
        and mean_intra < threshold * 0.5
    )
    metrics["chaining_detected"] = chaining_detected
    
    return metrics
