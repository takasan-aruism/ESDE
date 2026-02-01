"""
ESDE Phase 9: Edge Policy Resolver
====================================

Resolves the k parameter for Mutual-kNN edge selection by
sweeping candidate values and selecting based on observation
constraints.

k is not a free parameter — it is the lens's focal length.
  - k large → wide angle (global template patterns visible)
  - k small → telephoto (local thematic clusters visible)

The resolver does NOT "decide" the best k. It observes clustering
behavior at each k and selects the SMALLEST k that satisfies
descriptive constraints (giant_ratio, mean_intra_similarity).

"Describe, but do not decide" — k is chosen by what the data shows,
not by what we want it to show.

Spec: Phase 9 Edge Policy v1.0
"""

import math
import csv
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .edge_selector import MutualKNNSelector
from .w5_w6_adapter import SimpleCondensator
from .chaining_metrics import compute_chaining_metrics


EDGE_POLICY_VERSION = "v1.0"

# Default sweep candidates: geometric-ish progression
DEFAULT_K_CANDIDATES = [2, 3, 4, 5, 7, 9, 12, 15]


@dataclass
class SweepRow:
    """One row in the k-sweep results."""
    k: int
    n_edges: int
    n_islands: int
    n_noise: int
    largest_island: int
    giant_ratio: float
    edge_sparsity: float
    mean_intra_sim: Optional[float]
    chaining_detected: bool
    satisfies_policy: bool


@dataclass
class PolicyResult:
    """Output of EdgePolicyResolver.resolve()."""
    k_chosen: int
    structure: Any  # SimpleStructure
    chaining: Dict[str, Any]
    edge_filter_trace: Dict[str, Any]
    sweep: List[SweepRow]
    policy_trace: Dict[str, Any]


class EdgePolicyResolver:
    """
    Resolves k for Mutual-kNN by sweeping candidates.
    
    Policy constraints (descriptive, not prescriptive):
      - max_giant_ratio: largest island must be ≤ this fraction of total
      - min_mean_intra: mean intra-similarity of largest island must be ≥ this
    
    Selection rule: SMALLEST k that satisfies ALL constraints.
    If no k satisfies, fall back to the k with the lowest giant_ratio.
    
    Attributes:
        max_giant_ratio: Upper bound on giant component ratio (default: 0.20)
        min_mean_intra: Lower bound on mean intra-similarity (default: 0.25)
        candidates: List of k values to sweep
    """
    
    def __init__(
        self,
        max_giant_ratio: float = 0.20,
        min_mean_intra: float = 0.25,
        candidates: Optional[List[int]] = None,
    ):
        self.max_giant_ratio = max_giant_ratio
        self.min_mean_intra = min_mean_intra
        self.candidates = candidates or DEFAULT_K_CANDIDATES
    
    def resolve(
        self,
        all_sim_tuples: List[Tuple[str, str, float]],
        threshold: float,
        node_ids: List[str],
        node_vectors: Dict[str, Dict[str, float]],
        min_island_size: int = 2,
    ) -> PolicyResult:
        """
        Sweep k candidates and select the best.
        
        The similarity computation (W4) is done ONCE. Only edge filtering
        and clustering are repeated per candidate — this is lightweight.
        
        Args:
            all_sim_tuples: [(a, b, sim), ...] sorted descending.
            threshold: Resolved similarity threshold.
            node_ids: All condition/node identifiers.
            node_vectors: {node_id: {dim: value}} for centroid computation.
            min_island_size: Minimum island size for clustering.
            
        Returns:
            PolicyResult with chosen k, structure, traces, and sweep data.
        """
        n_nodes = len(node_ids)
        sweep_results: List[SweepRow] = []
        
        # Store full results for each k so we can select later
        full_results: Dict[int, Dict[str, Any]] = {}
        
        for k in self.candidates:
            if k >= n_nodes:
                continue  # k larger than N is meaningless
            
            # 1. Edge selection
            selector = MutualKNNSelector(k=k)
            filtered_edges, ef_trace = selector.select(all_sim_tuples, threshold)
            
            # 2. Clustering
            condensator = SimpleCondensator(
                threshold=threshold, min_island_size=min_island_size,
            )
            structure = condensator.condense_from_edges(
                node_ids=node_ids,
                node_vectors=node_vectors,
                edges=filtered_edges,
            )
            
            # 3. Chaining metrics
            island_dicts = [
                {"member_ids": isl.member_ids, "size": isl.size}
                for isl in structure.islands
            ]
            chaining = compute_chaining_metrics(
                islands=island_dicts,
                noise_ids=structure.noise_ids,
                input_count=structure.input_count,
                all_sim_pairs=all_sim_tuples,
                threshold=threshold,
            )
            
            # 4. Check policy
            gcr = chaining["giant_component_ratio"]
            mis = chaining["mean_intra_similarity"]
            satisfies = (
                gcr <= self.max_giant_ratio
                and (mis is None or mis >= self.min_mean_intra)
            )
            
            row = SweepRow(
                k=k,
                n_edges=ef_trace["n_mutual_knn"],
                n_islands=structure.island_count,
                n_noise=structure.noise_count,
                largest_island=chaining["largest_island_size"],
                giant_ratio=gcr,
                edge_sparsity=chaining["edge_sparsity"],
                mean_intra_sim=mis,
                chaining_detected=chaining["chaining_detected"],
                satisfies_policy=satisfies,
            )
            sweep_results.append(row)
            
            full_results[k] = {
                "structure": structure,
                "chaining": chaining,
                "edge_filter_trace": ef_trace,
            }
        
        # Selection: smallest k that satisfies policy
        satisfying = [r for r in sweep_results if r.satisfies_policy]
        
        if satisfying:
            chosen_k = satisfying[0].k  # First = smallest (candidates are sorted)
            selection_reason = "policy_satisfied"
        else:
            # Fallback: k with lowest giant_ratio
            best = min(sweep_results, key=lambda r: r.giant_ratio)
            chosen_k = best.k
            selection_reason = "fallback_min_giant_ratio"
        
        chosen = full_results[chosen_k]
        
        policy_trace = {
            "version": EDGE_POLICY_VERSION,
            "method": "k_sweep",
            "k_chosen": chosen_k,
            "selection_reason": selection_reason,
            "policy": {
                "max_giant_ratio": self.max_giant_ratio,
                "min_mean_intra": self.min_mean_intra,
            },
            "candidates_swept": [r.k for r in sweep_results],
            "n_satisfying": len(satisfying),
            "sweep_summary": [
                {
                    "k": r.k,
                    "islands": r.n_islands,
                    "noise": r.n_noise,
                    "largest": r.largest_island,
                    "gcr": r.giant_ratio,
                    "mean_intra": r.mean_intra_sim,
                    "satisfies": r.satisfies_policy,
                }
                for r in sweep_results
            ],
        }
        
        return PolicyResult(
            k_chosen=chosen_k,
            structure=chosen["structure"],
            chaining=chosen["chaining"],
            edge_filter_trace=chosen["edge_filter_trace"],
            sweep=sweep_results,
            policy_trace=policy_trace,
        )


def export_sweep_csv(
    sweep: List[SweepRow],
    output_path: str,
) -> str:
    """
    Export k-sweep results to CSV for analysis.
    
    Args:
        sweep: List of SweepRow from PolicyResult.
        output_path: Path to write CSV.
        
    Returns:
        Path written.
    """
    fieldnames = [
        "k", "n_edges", "n_islands", "n_noise", "largest_island",
        "giant_ratio", "edge_sparsity", "mean_intra_sim",
        "chaining_detected", "satisfies_policy",
    ]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sweep:
            writer.writerow({
                "k": row.k,
                "n_edges": row.n_edges,
                "n_islands": row.n_islands,
                "n_noise": row.n_noise,
                "largest_island": row.largest_island,
                "giant_ratio": f"{row.giant_ratio:.4f}",
                "edge_sparsity": f"{row.edge_sparsity:.4f}",
                "mean_intra_sim": f"{row.mean_intra_sim:.4f}" if row.mean_intra_sim is not None else "",
                "chaining_detected": row.chaining_detected,
                "satisfies_policy": row.satisfies_policy,
            })
    
    return output_path
