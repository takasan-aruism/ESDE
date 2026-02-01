"""
ESDE Phase 9: Global Threshold Model
=======================================

The "absolute" threshold is not a magic number — it is what ESDE has
observed across ALL experiments for a given (lens, feature_mode) pair.

    t_abs = Q_global(q)

Where Q_global is the quantile of the accumulated similarity distribution.

This makes t_abs an "empirical universal": grounded in data, not decree.
When global data is insufficient (new lens, first run), a fallback is used
and recorded in the trace.

Storage:
    data/threshold/{key}.json

    Each file accumulates pairwise similarities from every pipeline run
    for that (lens, feature_mode) combination.

Design source:
    - Taka: "Why is the absolute value what it is? It should come from
             all data ESDE has seen."
    - GPT: "Absolute = global distribution model. Same quantile method,
            different scope."

Spec: Phase 9 GlobalThresholdModel v1.0
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


# Minimum pairs needed before global quantile is considered "sufficient"
# Below this, fallback is used and traced.
MIN_GLOBAL_PAIRS = 30


def _make_key(lens: str, feature_mode: str) -> str:
    """
    Build storage key from lens + feature_mode.
    
    Examples:
        structure_token, semantic_vector, hybrid_vector
    """
    return f"{lens}_{feature_mode}"


class GlobalThresholdModel:
    """
    Accumulates pairwise similarity distributions across pipeline runs.
    
    One model instance handles all (lens, feature_mode) combinations.
    Each combination is stored in a separate JSON file under data_dir.
    
    Usage:
        model = GlobalThresholdModel("./data/threshold")
        
        # Query (before appending current run)
        t_abs, info = model.get_threshold("semantic", "vector", q=0.98)
        
        # Append (after run completes)
        model.append("semantic", "vector", similarities, dataset="mixed")
    """
    
    def __init__(self, data_dir: str = "./data/threshold"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Cache loaded data in memory
        self._cache: Dict[str, Dict] = {}
    
    def _path_for(self, key: str) -> Path:
        return self.data_dir / f"{key}.json"
    
    def _load(self, key: str) -> Dict:
        """Load or initialize data for a key."""
        if key in self._cache:
            return self._cache[key]
        
        path = self._path_for(key)
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
        else:
            data = {
                "key": key,
                "similarities": [],
                "run_log": [],
            }
        
        self._cache[key] = data
        return data
    
    def _save(self, key: str):
        """Persist data for a key."""
        if key not in self._cache:
            return
        
        path = self._path_for(key)
        with open(path, 'w') as f:
            json.dump(self._cache[key], f, indent=2)
    
    # ------------------------------------------
    # Query (read-only, before current run)
    # ------------------------------------------
    
    def get_count(self, lens: str, feature_mode: str) -> int:
        """Number of accumulated similarity pairs."""
        key = _make_key(lens, feature_mode)
        data = self._load(key)
        return len(data["similarities"])
    
    def is_sufficient(self, lens: str, feature_mode: str) -> bool:
        """Whether global data is sufficient for quantile estimation."""
        return self.get_count(lens, feature_mode) >= MIN_GLOBAL_PAIRS
    
    def get_quantile(self, lens: str, feature_mode: str, q: float = 0.98) -> Optional[float]:
        """
        Compute quantile from accumulated global similarities.
        
        Returns None if insufficient data.
        """
        key = _make_key(lens, feature_mode)
        data = self._load(key)
        sims = data["similarities"]
        
        if len(sims) < MIN_GLOBAL_PAIRS:
            return None
        
        sorted_s = sorted(sims)
        n = len(sorted_s)
        pos = q * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return sorted_s[lo] * (1 - frac) + sorted_s[hi] * frac
    
    def get_summary(self, lens: str, feature_mode: str) -> Dict[str, Any]:
        """Get summary of accumulated global distribution."""
        import math
        
        key = _make_key(lens, feature_mode)
        data = self._load(key)
        sims = data["similarities"]
        
        if not sims:
            return {
                "n_global": 0,
                "sufficient": False,
                "run_count": len(data["run_log"]),
            }
        
        sorted_s = sorted(sims)
        n = len(sorted_s)
        mean = sum(sorted_s) / n
        variance = sum((x - mean) ** 2 for x in sorted_s) / n
        std = math.sqrt(variance)
        
        def qval(q):
            pos = q * (n - 1)
            lo = int(pos)
            hi = min(lo + 1, n - 1)
            frac = pos - lo
            return round(sorted_s[lo] * (1 - frac) + sorted_s[hi] * frac, 6)
        
        return {
            "n_global": n,
            "sufficient": n >= MIN_GLOBAL_PAIRS,
            "run_count": len(data["run_log"]),
            "min": round(sorted_s[0], 6),
            "max": round(sorted_s[-1], 6),
            "mean": round(mean, 6),
            "std": round(std, 6),
            "quantiles": {
                f"q{int(q*100):02d}": qval(q)
                for q in [0.50, 0.75, 0.90, 0.95, 0.98, 0.99]
            },
        }
    
    def get_threshold(
        self,
        lens: str,
        feature_mode: str,
        q: float = 0.98,
        fallback: float = 0.0,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Get absolute threshold from global model.
        
        Returns:
            (threshold, info) where info contains:
                - source: "global" or "fallback"
                - n_global: how many pairs in global pool
                - q_abs: quantile used
                - global_summary: distribution stats (if sufficient)
        """
        key = _make_key(lens, feature_mode)
        n = self.get_count(lens, feature_mode)
        
        if n >= MIN_GLOBAL_PAIRS:
            t_abs = self.get_quantile(lens, feature_mode, q)
            return t_abs, {
                "source": "global",
                "n_global": n,
                "q_abs": q,
                "global_summary": self.get_summary(lens, feature_mode),
            }
        else:
            return fallback, {
                "source": "fallback",
                "n_global": n,
                "fallback_value": fallback,
                "reason": f"insufficient global data ({n} < {MIN_GLOBAL_PAIRS} pairs)",
            }
    
    # ------------------------------------------
    # Append (after current run completes)
    # ------------------------------------------
    
    def append(
        self,
        lens: str,
        feature_mode: str,
        similarities: List[float],
        dataset: str = "unknown",
        axis: str = "unknown",
    ):
        """
        Append pairwise similarities from a completed pipeline run.
        
        Called AFTER threshold resolution, so current run doesn't
        influence its own threshold (avoids self-reference).
        """
        key = _make_key(lens, feature_mode)
        data = self._load(key)
        
        # Append similarities
        data["similarities"].extend(similarities)
        
        # Log the run
        data["run_log"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset,
            "axis": axis,
            "pairs_added": len(similarities),
            "total_after": len(data["similarities"]),
        })
        
        # Persist
        self._save(key)
    
    # ------------------------------------------
    # Utility
    # ------------------------------------------
    
    def list_keys(self) -> List[str]:
        """List all (lens, feature_mode) keys with stored data."""
        keys = []
        for path in self.data_dir.glob("*.json"):
            keys.append(path.stem)
        return sorted(keys)
    
    def reset(self, lens: str, feature_mode: str):
        """Clear accumulated data for a key (useful for testing)."""
        key = _make_key(lens, feature_mode)
        self._cache.pop(key, None)
        path = self._path_for(key)
        if path.exists():
            path.unlink()
