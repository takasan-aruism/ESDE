#!/usr/bin/env python3
"""
ESDE Phase 9: S-Score Calculator (W3)
======================================

Calculates S-Score (specificity score) for each token under each condition.

Mathematical Model:
  S(t, C) = P(t|C) * log((P(t|C) + ε) / (P(t|G) + ε))

Where:
  - P(t|C) = Token probability under condition C
  - P(t|G) = Token probability globally
  - ε = Smoothing constant (1e-12)

Interpretation:
  - S > 0: Token is MORE specific to condition (appears more than expected)
  - S < 0: Token is LESS specific to condition (appears less than expected)
  - S ≈ 0: Token appears at similar rate everywhere

Output:
  - W3Result: Per-condition top-K positive and negative candidates
  - Ready for W4 projection

Spec: Phase 9 W3 v1.0
"""

import math
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone

from .w2_aggregator import W2Stats, ConditionStats


# ==========================================
# Constants
# ==========================================

W3_VERSION = "v9.phase9.1"
W3_ALGORITHM = "KLContribution-v1"
EPSILON = 1e-12
DEFAULT_TOP_K = 100
DEFAULT_MIN_COUNT = 2


# ==========================================
# Data Structures
# ==========================================

@dataclass
class CandidateToken:
    """A token candidate with S-Score."""
    token: str
    s_score: float
    p_condition: float  # P(t|C)
    p_global: float     # P(t|G)
    count_condition: int
    count_global: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "token": self.token,
            "s_score": round(self.s_score, 8),
            "p_condition": round(self.p_condition, 8),
            "p_global": round(self.p_global, 8),
            "count_condition": self.count_condition,
            "count_global": self.count_global,
        }


@dataclass
class ConditionCandidates:
    """S-Score candidates for a single condition."""
    condition_id: str
    factors: Dict[str, Any]
    positive_candidates: List[CandidateToken] = field(default_factory=list)
    negative_candidates: List[CandidateToken] = field(default_factory=list)
    total_compared: int = 0
    positive_count: int = 0
    negative_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "factors": self.factors,
            "total_compared": self.total_compared,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "positive_candidates": [c.to_dict() for c in self.positive_candidates],
            "negative_candidates": [c.to_dict() for c in self.negative_candidates],
        }


@dataclass
class W3Result:
    """
    W3 Calculation result.
    
    Contains S-Score candidates for all conditions.
    """
    # Source info
    provider_id: str
    axis_name: str
    w2_snapshot_hash: str
    
    # Per-condition candidates
    conditions: Dict[str, ConditionCandidates] = field(default_factory=dict)
    
    # Parameters
    top_k: int = DEFAULT_TOP_K
    min_count: int = DEFAULT_MIN_COUNT
    epsilon: float = EPSILON
    algorithm: str = W3_ALGORITHM
    
    # Metadata
    version: str = W3_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "axis_name": self.axis_name,
            "w2_snapshot_hash": self.w2_snapshot_hash,
            "condition_count": len(self.conditions),
            "top_k": self.top_k,
            "min_count": self.min_count,
            "algorithm": self.algorithm,
            "version": self.version,
            "created_at": self.created_at,
            "conditions": {
                cid: cond.to_dict()
                for cid, cond in self.conditions.items()
            },
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def get_sscore_dict(self, condition_id: str) -> Dict[str, float]:
        """
        Get token → S-Score mapping for a condition.
        
        Used by W4 for projection.
        """
        if condition_id not in self.conditions:
            return {}
        
        cond = self.conditions[condition_id]
        sscore_dict = {}
        
        for c in cond.positive_candidates:
            sscore_dict[c.token] = c.s_score
        
        for c in cond.negative_candidates:
            sscore_dict[c.token] = c.s_score
        
        return sscore_dict


# ==========================================
# W3 Calculator
# ==========================================

class W3Calculator:
    """
    Calculates S-Score candidates from W2 statistics.
    
    Usage:
        calculator = W3Calculator(w2_stats, top_k=100)
        result = calculator.calculate_all()
        
        # Get S-Score dict for projection
        sscore_dict = result.get_sscore_dict("Lead")
    """
    
    def __init__(
        self,
        w2_stats: W2Stats,
        top_k: int = DEFAULT_TOP_K,
        min_count: int = DEFAULT_MIN_COUNT,
        epsilon: float = EPSILON,
    ):
        """
        Initialize W3 Calculator.
        
        Args:
            w2_stats: W2 statistics
            top_k: Number of top candidates per condition
            min_count: Minimum count to consider token
            epsilon: Smoothing constant
        """
        self.w2_stats = w2_stats
        self.top_k = top_k
        self.min_count = min_count
        self.epsilon = epsilon
        
        # Compute snapshot hash for reproducibility
        self._w2_hash = self._compute_w2_hash()
    
    def _compute_w2_hash(self) -> str:
        """Compute hash of W2 statistics for traceability."""
        # Use a subset of data for hash (not full token counts)
        hash_data = {
            "provider_id": self.w2_stats.provider_id,
            "global_total": self.w2_stats.global_total,
            "unique_tokens": len(self.w2_stats.global_counts),
            "condition_count": len(self.w2_stats.conditions),
            "articles_processed": self.w2_stats.articles_processed,
        }
        json_str = json.dumps(hash_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
    
    def _compute_s_score(
        self,
        count_cond: int,
        total_cond: int,
        count_global: int,
        total_global: int,
    ) -> Tuple[float, float, float]:
        """
        Compute S-Score (per-token KL contribution).
        
        S(t, C) = P(t|C) * log((P(t|C) + ε) / (P(t|G) + ε))
        
        Returns:
            (s_score, p_condition, p_global)
        """
        # Compute probabilities
        p_cond = count_cond / total_cond if total_cond > 0 else 0.0
        p_global = count_global / total_global if total_global > 0 else 0.0
        
        # Apply smoothing
        p_cond_smoothed = p_cond + self.epsilon
        p_global_smoothed = p_global + self.epsilon
        
        # Compute S-Score
        s_score = p_cond * math.log(p_cond_smoothed / p_global_smoothed)
        
        return s_score, p_cond, p_global
    
    def calculate_condition(self, condition_id: str) -> ConditionCandidates:
        """
        Calculate S-Score candidates for a single condition.
        
        Args:
            condition_id: Condition identifier
            
        Returns:
            ConditionCandidates with positive and negative lists
        """
        if condition_id not in self.w2_stats.conditions:
            raise ValueError(f"Unknown condition: {condition_id}")
        
        cond_stats = self.w2_stats.conditions[condition_id]
        global_total = self.w2_stats.global_total
        cond_total = cond_stats.total_tokens
        
        if cond_total == 0:
            return ConditionCandidates(
                condition_id=condition_id,
                factors=cond_stats.factors,
            )
        
        # Collect all tokens to compare
        all_tokens = set(cond_stats.token_counts.keys()) | set(self.w2_stats.global_counts.keys())
        
        scored_tokens: List[Tuple[str, float, float, float, int, int]] = []
        
        for token in all_tokens:
            count_cond = cond_stats.token_counts.get(token, 0)
            count_global = self.w2_stats.global_counts.get(token, 0)
            
            # Filter by min_count
            if count_cond < self.min_count and count_global < self.min_count:
                continue
            
            s_score, p_cond, p_global = self._compute_s_score(
                count_cond, cond_total,
                count_global, global_total,
            )
            
            scored_tokens.append((
                token, s_score, p_cond, p_global, count_cond, count_global
            ))
        
        # Sort: positive (highest first), negative (lowest first)
        positive_tokens = [t for t in scored_tokens if t[1] > 0]
        positive_tokens.sort(key=lambda x: (-x[1], x[0]))  # Desc by score, asc by name
        
        negative_tokens = [t for t in scored_tokens if t[1] < 0]
        negative_tokens.sort(key=lambda x: (x[1], x[0]))  # Asc by score (most negative first)
        
        # Build candidate lists
        positive_candidates = [
            CandidateToken(
                token=t[0],
                s_score=t[1],
                p_condition=t[2],
                p_global=t[3],
                count_condition=t[4],
                count_global=t[5],
            )
            for t in positive_tokens[:self.top_k]
        ]
        
        negative_candidates = [
            CandidateToken(
                token=t[0],
                s_score=t[1],
                p_condition=t[2],
                p_global=t[3],
                count_condition=t[4],
                count_global=t[5],
            )
            for t in negative_tokens[:self.top_k]
        ]
        
        return ConditionCandidates(
            condition_id=condition_id,
            factors=cond_stats.factors,
            positive_candidates=positive_candidates,
            negative_candidates=negative_candidates,
            total_compared=len(scored_tokens),
            positive_count=len(positive_tokens),
            negative_count=len(negative_tokens),
        )
    
    def calculate_all(self) -> W3Result:
        """
        Calculate S-Score candidates for all conditions.
        
        Returns:
            W3Result with all conditions
        """
        result = W3Result(
            provider_id=self.w2_stats.provider_id,
            axis_name=self.w2_stats.axis_name,
            w2_snapshot_hash=self._w2_hash,
            top_k=self.top_k,
            min_count=self.min_count,
            epsilon=self.epsilon,
        )
        
        for condition_id in self.w2_stats.conditions.keys():
            try:
                candidates = self.calculate_condition(condition_id)
                result.conditions[condition_id] = candidates
            except Exception as e:
                print(f"[W3Calculator] Error for {condition_id}: {e}")
        
        return result
    
    def get_summary(self, result: W3Result) -> Dict[str, Any]:
        """Get human-readable summary of W3 result."""
        return {
            "provider_id": result.provider_id,
            "axis_name": result.axis_name,
            "condition_count": len(result.conditions),
            "conditions": [
                {
                    "condition_id": cid,
                    "positive_count": cond.positive_count,
                    "negative_count": cond.negative_count,
                    "top_positive": [
                        (c.token, round(c.s_score, 6))
                        for c in cond.positive_candidates[:5]
                    ],
                    "top_negative": [
                        (c.token, round(c.s_score, 6))
                        for c in cond.negative_candidates[:5]
                    ],
                }
                for cid, cond in sorted(
                    result.conditions.items(),
                    key=lambda x: -x[1].total_compared
                )
            ],
        }


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("W3 Calculator Test")
    print("=" * 60)
    
    from .w2_aggregator import W2Stats, ConditionStats
    from collections import defaultdict
    
    # Create mock W2 stats
    w2 = W2Stats(
        provider_id="section_v1",
        axis_name="section",
        aggregation_unit="section",
    )
    
    # Global counts
    w2.global_counts = defaultdict(int, {
        "the": 100,
        "war": 50,
        "peace": 20,
        "battle": 30,
        "art": 15,
        "culture": 10,
    })
    w2.global_total = sum(w2.global_counts.values())
    
    # Condition 1: Military
    military = ConditionStats(
        condition_id="military",
        factors={"section_name": "military"},
    )
    military.token_counts = defaultdict(int, {
        "the": 40,
        "war": 35,  # High in military
        "battle": 25,  # High in military
        "peace": 2,  # Low in military
    })
    military.total_tokens = sum(military.token_counts.values())
    w2.conditions["military"] = military
    
    # Condition 2: Culture
    culture = ConditionStats(
        condition_id="culture",
        factors={"section_name": "culture"},
    )
    culture.token_counts = defaultdict(int, {
        "the": 30,
        "art": 12,  # High in culture
        "culture": 8,  # High in culture
        "war": 2,  # Low in culture
    })
    culture.total_tokens = sum(culture.token_counts.values())
    w2.conditions["culture"] = culture
    
    # Calculate
    calculator = W3Calculator(w2, top_k=10, min_count=2)
    result = calculator.calculate_all()
    
    # Print results
    summary = calculator.get_summary(result)
    
    print(f"\n[Results]")
    print(f"  Axis: {summary['axis_name']}")
    print(f"  Conditions: {summary['condition_count']}")
    
    for cond in summary['conditions']:
        print(f"\n  [{cond['condition_id']}]")
        print(f"    Top positive (specific to this condition):")
        for token, score in cond['top_positive']:
            print(f"      {token}: {score:+.6f}")
        print(f"    Top negative (less common in this condition):")
        for token, score in cond['top_negative']:
            print(f"      {token}: {score:+.6f}")
    
    print("\n" + "=" * 60)
    print("W3 Calculator tests passed!")
