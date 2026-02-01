"""
ESDE Phase 9: Edge Selector
============================

Filters similarity edges using Mutual k-Nearest Neighbor criterion.
Prevents single-linkage chaining by requiring bidirectional affinity.

Single-linkage clustering connects nodes through transitive chains:
  A-B (0.70) + B-C (0.68) + C-D (0.66) → A,B,C,D in one island
even when A↔D similarity is only 0.30.

Mutual-kNN breaks these chains: an edge (i,j) is kept only if
  j is among i's top-k neighbors AND i is among j's top-k neighbors.
One-sided affinity is not enough.

"Describe, but do not decide" — the selector observes mutual proximity,
it does not impose structure.

Spec: Phase 9 Edge Selector v1.0
"""

import math
from typing import List, Dict, Any, Optional, Tuple, Set


EDGE_SELECTOR_VERSION = "v1.0"


class MutualKNNSelector:
    """
    Mutual k-Nearest Neighbor edge selector.
    
    For each node, identifies its k most similar neighbors.
    An edge is kept only if BOTH endpoints consider the other
    a top-k neighbor. This is the "mutual" criterion.
    
    k can be fixed or dynamic (ceil(log2(N))).
    
    Attributes:
        k: Number of nearest neighbors. None = auto.
    """
    
    def __init__(self, k: Optional[int] = None):
        """
        Args:
            k: Number of nearest neighbors per node.
               None → auto: ceil(log2(N)) where N is number of nodes.
        """
        self.k = k
    
    def select(
        self,
        sim_pairs: List[Tuple[str, str, float]],
        threshold: float,
    ) -> Tuple[List[Tuple[str, str, float]], Dict[str, Any]]:
        """
        Filter similarity pairs using mutual k-nearest neighbors.
        
        An edge (i, j) is kept if and only if:
          1. j is in top-k neighbors of i  (based on ALL pairs)
          2. i is in top-k neighbors of j  (mutual requirement)
          3. sim(i, j) >= threshold         (quality floor)
        
        Top-k is computed from the FULL similarity landscape (before
        thresholding) so neighborhood structure reflects the complete
        distribution, not just the above-threshold subset.
        
        Args:
            sim_pairs: [(a, b, similarity), ...] sorted descending.
            threshold: Minimum similarity for edge adoption.
            
        Returns:
            (filtered_pairs, trace_info)
            
            filtered_pairs: [(a, b, sim), ...] that passed all criteria.
            trace_info: Diagnostic dict for Substrate recording.
        """
        if not sim_pairs:
            return [], self._empty_trace()
        
        # Collect all node IDs
        node_ids: Set[str] = set()
        for a, b, _ in sim_pairs:
            node_ids.add(a)
            node_ids.add(b)
        
        n = len(node_ids)
        k = self._resolve_k(n)
        
        # Build top-k neighbors for each node.
        # sim_pairs is sorted desc, so first encounters are highest similarity.
        topk: Dict[str, List[str]] = {nid: [] for nid in node_ids}
        
        for a, b, sim in sim_pairs:
            if len(topk[a]) < k:
                topk[a].append(b)
            if len(topk[b]) < k:
                topk[b].append(a)
        
        # Convert to sets for O(1) lookup
        topk_sets: Dict[str, Set[str]] = {
            nid: set(neighbors) for nid, neighbors in topk.items()
        }
        
        # Filter: mutual-kNN AND above threshold
        # sim_pairs is sorted desc, so we can break early
        n_above_threshold = 0
        filtered: List[Tuple[str, str, float]] = []
        
        for a, b, sim in sim_pairs:
            if sim < threshold:
                break  # sorted desc → all remaining are below
            n_above_threshold += 1
            
            if b in topk_sets[a] and a in topk_sets[b]:
                filtered.append((a, b, sim))
        
        # Build trace
        trace = {
            "version": EDGE_SELECTOR_VERSION,
            "method": "mutual_knn",
            "k": k,
            "k_auto": self.k is None,
            "n_nodes": n,
            "n_total_pairs": len(sim_pairs),
            "n_above_threshold": n_above_threshold,
            "n_mutual_knn": len(filtered),
            "reduction_ratio": round(
                1.0 - (len(filtered) / max(1, n_above_threshold)),
                4,
            ),
        }
        
        return filtered, trace
    
    def _resolve_k(self, n: int) -> int:
        """Determine k: explicit or auto (ceil(log2(N)))."""
        if self.k is not None:
            return self.k
        return max(1, math.ceil(math.log2(max(2, n))))
    
    def _empty_trace(self) -> Dict[str, Any]:
        """Trace for empty input."""
        return {
            "version": EDGE_SELECTOR_VERSION,
            "method": "mutual_knn",
            "k": self.k or 0,
            "k_auto": self.k is None,
            "n_nodes": 0,
            "n_total_pairs": 0,
            "n_above_threshold": 0,
            "n_mutual_knn": 0,
            "reduction_ratio": 0.0,
        }


class NoOpSelector:
    """
    Pass-through selector. Keeps all edges above threshold.
    
    Used when edge filtering is disabled (--edge-filter none).
    Provides trace for consistency.
    """
    
    def select(
        self,
        sim_pairs: List[Tuple[str, str, float]],
        threshold: float,
    ) -> Tuple[List[Tuple[str, str, float]], Dict[str, Any]]:
        """Keep all pairs above threshold."""
        filtered = [(a, b, sim) for a, b, sim in sim_pairs if sim >= threshold]
        
        trace = {
            "version": EDGE_SELECTOR_VERSION,
            "method": "none",
            "n_nodes": len(set(a for a, _, _ in sim_pairs) | set(b for _, b, _ in sim_pairs)),
            "n_total_pairs": len(sim_pairs),
            "n_above_threshold": len(filtered),
            "n_mutual_knn": len(filtered),
            "reduction_ratio": 0.0,
        }
        
        return filtered, trace


def create_edge_selector(
    method: str = "none",
    k: Optional[int] = None,
) -> Any:
    """
    Factory for edge selectors.
    
    Args:
        method: "none" (pass-through) or "mutual_knn"
        k: For mutual_knn, the k parameter. None = auto.
        
    Returns:
        Selector instance with .select() method.
    """
    if method == "mutual_knn":
        return MutualKNNSelector(k=k)
    elif method == "none":
        return NoOpSelector()
    else:
        raise ValueError(f"Unknown edge selector method: {method}")
