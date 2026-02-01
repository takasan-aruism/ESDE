"""
ESDE Phase 9: Threshold Resolver
==================================

Dynamic threshold as a **descriptor**, not a decider.

The resolver observes the similarity distribution and records what threshold
emerges from the data. It does not "decide" — it describes.

Design source:
  - GPT: Absolute + Relative + Resolve (Three-term emergence)
  - Taka: "Describe, but do not decide" applied to thresholds
  - Claude: Resolver is a descriptor, all values traced

Modes:
  - fixed: Use t_abs directly (legacy behavior)
  - quantile: t_rel = quantile(S, q), t = max(t_abs_floor, t_rel)

Future extensions (plug-in at relative side):
  - knn: k-nearest neighbor gap detection
  - optimize: Objective function sweep

Spec: Phase 9 Threshold v1.0
"""

import math
from typing import List, Dict, Any, Optional


THRESHOLD_VERSION = "v1.1"  # v1.1: global model support


# ==========================================
# Distribution Summary
# ==========================================

def summarize_distribution(sims: List[float]) -> Dict[str, Any]:
    """
    Compute descriptive statistics of pairwise similarities.
    
    This is a pure observation — no decisions made here.
    All values are recorded for trace/audit.
    
    Args:
        sims: List of pairwise similarity values
        
    Returns:
        Dictionary with distribution summary
    """
    if not sims:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "quantiles": {},
        }
    
    n = len(sims)
    sorted_s = sorted(sims)
    
    mean = sum(sorted_s) / n
    variance = sum((x - mean) ** 2 for x in sorted_s) / n
    std = math.sqrt(variance)
    
    def quantile_val(q: float) -> float:
        """Compute quantile using linear interpolation."""
        pos = q * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return sorted_s[lo] * (1 - frac) + sorted_s[hi] * frac
    
    quantiles = {}
    for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98, 0.99]:
        quantiles[f"q{int(q*100):02d}"] = round(quantile_val(q), 6)
    
    return {
        "count": n,
        "min": round(sorted_s[0], 6),
        "max": round(sorted_s[-1], 6),
        "mean": round(mean, 6),
        "std": round(std, 6),
        "quantiles": quantiles,
    }


# ==========================================
# Relative Threshold: Quantile
# ==========================================

def relative_threshold_quantile(sims: List[float], q: float = 0.98) -> float:
    """
    Compute relative threshold from similarity distribution quantile.
    
    "What does 'similar' mean in this dataset?"
    → The top (1-q)% of pairs are considered 'similar'.
    
    Args:
        sims: List of pairwise similarity values
        q: Quantile (0.0 to 1.0). Default 0.98 = top 2%
        
    Returns:
        Threshold value at the given quantile
    """
    if not sims:
        return 0.0
    
    sorted_s = sorted(sims)
    n = len(sorted_s)
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    
    return sorted_s[lo] * (1 - frac) + sorted_s[hi] * frac


# ==========================================
# Resolve: Compose absolute + relative
# ==========================================

def resolve_threshold(
    t_abs: float,
    t_rel: float,
    mode: str = "safety_first",
) -> float:
    """
    Compose absolute and relative thresholds.
    
    This is the "third term" — emergence from two observations.
    
    Modes:
      - safety_first: t = max(t_abs, t_rel)
        "Never go below the floor, but let data raise the bar"
        
    Future modes:
      - blend: t = α * t_rel + (1-α) * t_abs
      - two_pass: Use t_rel for candidates, t_abs for validation
    
    Args:
        t_abs: Absolute threshold floor
        t_rel: Relative threshold from data
        mode: Resolution strategy
        
    Returns:
        Resolved threshold
    """
    if mode == "safety_first":
        return max(t_abs, t_rel)
    else:
        raise ValueError(f"Unknown resolve mode: '{mode}'. Available: safety_first")


# ==========================================
# ThresholdTrace: Full record for Substrate
# ==========================================

def build_threshold_trace(
    t_abs: float,
    t_rel: Optional[float],
    t_resolved: float,
    mode: str,
    resolve_strategy: str,
    dist_summary: Dict[str, Any],
    quantile_q: Optional[float] = None,
    lens: Optional[str] = None,
    axis: Optional[str] = None,
    feature_mode: Optional[str] = None,
    abs_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a complete threshold trace record.
    
    This is the "observation report" — what the resolver saw and computed.
    Designed to be appended to Substrate as a machine-observable trace.
    
    Args:
        t_abs: Absolute floor used
        t_rel: Relative threshold computed (None if mode=fixed)
        t_resolved: Final threshold used for W5
        mode: Threshold mode ('fixed' or 'quantile')
        resolve_strategy: How abs+rel were combined ('safety_first')
        dist_summary: Similarity distribution statistics (current run)
        quantile_q: Quantile parameter used (if mode=quantile)
        lens: Lens name
        axis: Condition axis
        feature_mode: Feature mode (token/vector)
        abs_info: Global model info for t_abs derivation
        
    Returns:
        Complete trace dictionary
    """
    trace = {
        "version": THRESHOLD_VERSION,
        "mode": mode,
        "resolve_strategy": resolve_strategy,
        "t_abs": round(t_abs, 6),
        "t_rel": round(t_rel, 6) if t_rel is not None else None,
        "t_resolved": round(t_resolved, 6),
        "run_distribution": dist_summary,
    }
    
    if quantile_q is not None:
        trace["quantile_q"] = quantile_q
    
    if lens:
        trace["lens"] = lens
    if axis:
        trace["axis"] = axis
    if feature_mode:
        trace["feature_mode"] = feature_mode
    
    # Global model info: how t_abs was derived
    if abs_info:
        trace["abs_source"] = abs_info
    
    return trace
