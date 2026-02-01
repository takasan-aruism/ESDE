"""
ESDE Phase 9: W3 Vector Score
===============================

Vector-mode W3: Computes condition profiles as deviations from global mean.

Unlike token-mode W3 (S-Score = log frequency ratio), this computes:
  - Δ_c = μ_c - μ_global   (raw difference per dimension)
  - z_c[d] = (μ_c[d] - μ_global[d]) / (σ_global[d] + ε)  (z-score)

Design source:
  - GPT audit: "W3 = explanatory diff display, W4 = spatial distance"
  - GPT audit: "μ_global must be weighted average (Σ vec_sum / Σ token_count)"

Input:  W2Stats with vector_sum/vector_count per condition
Output: VectorProfile per condition (delta, z-score, top dimensions)

Spec: Phase 9 W3 Vector v1.0
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .w2_aggregator import W2Stats


# ==========================================
# Constants
# ==========================================

W3_VECTOR_VERSION = "v1.0"
EPSILON = 1e-9  # For z-score stability
VECTOR_DIM_NAMES = [
    "word_length", "syllable_count", "is_stopword", "has_uppercase",
    "is_all_caps", "has_hyphen", "is_numeric", "is_passive_participle",
    "has_prefix", "has_suffix", "sentence_position_norm",
    "concreteness", "aoa", "sensorimotor", "is_proper_noun",
    "valence", "arousal", "dominance",
    "in_parentheses", "in_quotes",
]


# ==========================================
# Data Structures
# ==========================================

@dataclass
class VectorProfile:
    """Profile for a single condition."""
    condition_id: str
    mean_vector: List[float]       # μ_c
    delta_vector: List[float]      # μ_c - μ_global
    z_score_vector: List[float]    # (μ_c - μ_global) / σ_global
    token_count: int
    
    # Top deviating dimensions (for human-readable output)
    top_positive: List[Dict[str, Any]] = field(default_factory=list)
    top_negative: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class W3VectorResult:
    """Complete W3 vector analysis result."""
    global_mean: List[float]
    global_std: List[float]
    profiles: Dict[str, VectorProfile]
    condition_count: int
    total_tokens: int
    version: str = W3_VECTOR_VERSION


# ==========================================
# W3 Vector Calculator
# ==========================================

class W3VectorCalculator:
    """
    Computes vector profiles from W2 stats.
    
    W3 role in vector mode: "Explain HOW each condition differs from average."
    This is for interpretability, not for clustering (that's W4).
    """
    
    def __init__(self, top_n: int = 5):
        self.top_n = top_n
    
    def calculate(self, w2_stats: W2Stats) -> W3VectorResult:
        """
        Compute vector profiles for all conditions.
        
        Args:
            w2_stats: W2Stats with vector accumulation data
            
        Returns:
            W3VectorResult with per-condition profiles
        """
        # 1. Compute global mean (weighted)
        global_mean = w2_stats.get_global_mean_vector()
        if global_mean is None:
            raise ValueError("No vector data in W2Stats. Was feature_mode='vector'?")
        
        ndim = len(global_mean)
        
        # 2. Compute global std from condition means
        condition_means = {}
        condition_counts = {}
        
        for cid, cstats in w2_stats.conditions.items():
            mean = cstats.get_mean_vector()
            if mean is not None:
                condition_means[cid] = mean
                condition_counts[cid] = cstats.vector_count
        
        # Weighted variance: Σ_c (count_c * (μ_c - μ_g)²) / Σ_c count_c
        global_std = self._compute_global_std(
            condition_means, condition_counts, global_mean, ndim
        )
        
        # 3. Compute profiles
        profiles = {}
        for cid, mean_c in condition_means.items():
            delta = [mean_c[d] - global_mean[d] for d in range(ndim)]
            z_score = [
                (mean_c[d] - global_mean[d]) / (global_std[d] + EPSILON)
                for d in range(ndim)
            ]
            
            # Top deviating dimensions
            dim_scores = [
                (d, VECTOR_DIM_NAMES[d] if d < len(VECTOR_DIM_NAMES) else f"dim_{d}",
                 z_score[d])
                for d in range(ndim)
            ]
            dim_scores.sort(key=lambda x: x[2], reverse=True)
            
            top_positive = [
                {"dim": idx, "name": name, "z_score": round(z, 4), "delta": round(delta[idx], 6)}
                for idx, name, z in dim_scores[:self.top_n]
                if z > 0
            ]
            top_negative = [
                {"dim": idx, "name": name, "z_score": round(z, 4), "delta": round(delta[idx], 6)}
                for idx, name, z in dim_scores[-self.top_n:]
                if z < 0
            ]
            top_negative.reverse()
            
            profiles[cid] = VectorProfile(
                condition_id=cid,
                mean_vector=[round(v, 6) for v in mean_c],
                delta_vector=[round(v, 6) for v in delta],
                z_score_vector=[round(v, 4) for v in z_score],
                token_count=condition_counts[cid],
                top_positive=top_positive,
                top_negative=top_negative,
            )
        
        return W3VectorResult(
            global_mean=[round(v, 6) for v in global_mean],
            global_std=[round(v, 6) for v in global_std],
            profiles=profiles,
            condition_count=len(profiles),
            total_tokens=w2_stats.global_vector_count,
        )
    
    def _compute_global_std(
        self,
        condition_means: Dict[str, List[float]],
        condition_counts: Dict[str, int],
        global_mean: List[float],
        ndim: int,
    ) -> List[float]:
        """
        Compute weighted standard deviation across conditions.
        
        Uses: σ²[d] = Σ_c (count_c * (μ_c[d] - μ_g[d])²) / Σ_c count_c
        """
        total_count = sum(condition_counts.values())
        if total_count == 0:
            return [0.0] * ndim
        
        variance = [0.0] * ndim
        for cid, mean_c in condition_means.items():
            count_c = condition_counts[cid]
            for d in range(ndim):
                diff = mean_c[d] - global_mean[d]
                variance[d] += count_c * diff * diff
        
        return [math.sqrt(variance[d] / total_count) for d in range(ndim)]


# ==========================================
# Export Helpers
# ==========================================

def export_vector_profiles(result: W3VectorResult) -> Dict[str, Any]:
    """Export W3 vector result to JSON-serializable dict."""
    return {
        "version": result.version,
        "condition_count": result.condition_count,
        "total_tokens": result.total_tokens,
        "global_mean": result.global_mean,
        "global_std": result.global_std,
        "dim_names": VECTOR_DIM_NAMES,
        "profiles": {
            cid: {
                "mean_vector": p.mean_vector,
                "delta_vector": p.delta_vector,
                "z_score_vector": p.z_score_vector,
                "token_count": p.token_count,
                "top_positive": p.top_positive,
                "top_negative": p.top_negative,
            }
            for cid, p in result.profiles.items()
        },
    }
