"""
ESDE Phase 9-3: W3 Schema
=========================
W3 (Weak Axis Candidates) data structures.

Design Principles:
  - W3 = Difference between W1 (global) and W2 (conditional)
  - Produces "axis candidates" (shadow of structure), NOT confirmed axes
  - No labeling, no interpretation (INV-W3-001)

Mathematical Model:
  S(t, C) = P(t|C) * log((P(t|C) + ε) / (P(t|G) + ε))
  
  This is the per-token KL contribution (specificity score), not full KL divergence.

Invariants:
  INV-W3-001 (No Labeling):
      Extracted tokens must not be labeled (e.g., "political axis").
      Only factual output: "tokens specific to condition X".
  
  INV-W3-002 (Immutable Input):
      W3 calculation must not modify W1/W2 data.
  
  INV-W3-003 (Deterministic):
      Same W1/W2 input must produce identical ranking and scores.

Spec: v5.4.3-P9.3
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import json
import hashlib


# ==========================================
# Constants
# ==========================================

W3_VERSION = "v9.3.0"
W3_ALGORITHM = "KL-PerToken-v1"

# P0-2: Fixed epsilon for smoothing
EPSILON = 1e-12

# P1-2: Minimum count filter for stability
DEFAULT_MIN_COUNT_FOR_W3 = 2

# P1-3: Default Top-K
DEFAULT_TOP_K = 100


# ==========================================
# Candidate Token
# ==========================================

@dataclass
class CandidateToken:
    """
    A token identified as an axis candidate.
    
    Contains the token and its specificity score.
    """
    token_norm: str             # Normalized token
    s_score: float              # Specificity Score (per-token KL contribution)
    p_cond: float               # P(t|C) - conditional probability
    p_global: float             # P(t|G) - global probability
    count_cond: int = 0         # Raw count under condition
    count_global: int = 0       # Raw count globally
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_norm": self.token_norm,
            "s_score": round(self.s_score, 8),
            "p_cond": round(self.p_cond, 10),
            "p_global": round(self.p_global, 10),
            "count_cond": self.count_cond,
            "count_global": self.count_global,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CandidateToken':
        return cls(
            token_norm=data["token_norm"],
            s_score=data["s_score"],
            p_cond=data["p_cond"],
            p_global=data["p_global"],
            count_cond=data.get("count_cond", 0),
            count_global=data.get("count_global", 0),
        )


# ==========================================
# W3 Record
# ==========================================

@dataclass
class W3Record:
    """
    W3: Axis candidates for a specific condition.
    
    Contains the top-K positive and negative candidates
    that distinguish this condition from the global distribution.
    
    INV-W3-001: No labels. Only "tokens specific to this condition".
    """
    
    # --- Identity ---
    # P1-5: Hash-based ID for reproducibility
    analysis_id: str                    # SHA256(cond_sig + w1_hash + w2_hash + algo)
    timestamp: str = ""                 # ISO8601
    
    # --- Target ---
    condition_signature: str = ""       # Target W2 condition
    condition_factors: Dict[str, str] = field(default_factory=dict)
    
    # --- Input Versions (for reproducibility) ---
    w1_snapshot_hash: str = ""          # Hash of W1 state
    w2_snapshot_hash: str = ""          # Hash of W2 state
    
    # --- The Candidates (The Shadow) ---
    # Positive: tokens MORE specific to this condition (S > 0, sorted desc)
    positive_candidates: List[CandidateToken] = field(default_factory=list)
    
    # Negative: tokens LESS specific to this condition (S < 0, sorted by S asc = most negative first)
    negative_candidates: List[CandidateToken] = field(default_factory=list)
    
    # --- Configuration ---
    algorithm: str = W3_ALGORITHM
    top_k: int = DEFAULT_TOP_K
    min_count: int = DEFAULT_MIN_COUNT_FOR_W3
    epsilon: float = EPSILON
    
    # --- Stats ---
    total_tokens_compared: int = 0
    positive_count: int = 0             # Total tokens with S > 0
    negative_count: int = 0             # Total tokens with S < 0
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "timestamp": self.timestamp,
            "condition_signature": self.condition_signature,
            "condition_factors": self.condition_factors,
            "w1_snapshot_hash": self.w1_snapshot_hash,
            "w2_snapshot_hash": self.w2_snapshot_hash,
            "positive_candidates": [c.to_dict() for c in self.positive_candidates],
            "negative_candidates": [c.to_dict() for c in self.negative_candidates],
            "algorithm": self.algorithm,
            "top_k": self.top_k,
            "min_count": self.min_count,
            "epsilon": self.epsilon,
            "total_tokens_compared": self.total_tokens_compared,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, sort_keys=True)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'W3Record':
        record = cls(
            analysis_id=data["analysis_id"],
            timestamp=data.get("timestamp", ""),
            condition_signature=data.get("condition_signature", ""),
            condition_factors=data.get("condition_factors", {}),
            w1_snapshot_hash=data.get("w1_snapshot_hash", ""),
            w2_snapshot_hash=data.get("w2_snapshot_hash", ""),
            algorithm=data.get("algorithm", W3_ALGORITHM),
            top_k=data.get("top_k", DEFAULT_TOP_K),
            min_count=data.get("min_count", DEFAULT_MIN_COUNT_FOR_W3),
            epsilon=data.get("epsilon", EPSILON),
            total_tokens_compared=data.get("total_tokens_compared", 0),
            positive_count=data.get("positive_count", 0),
            negative_count=data.get("negative_count", 0),
        )
        record.positive_candidates = [
            CandidateToken.from_dict(c) for c in data.get("positive_candidates", [])
        ]
        record.negative_candidates = [
            CandidateToken.from_dict(c) for c in data.get("negative_candidates", [])
        ]
        return record
    
    @classmethod
    def from_json(cls, json_str: str) -> 'W3Record':
        return cls.from_dict(json.loads(json_str))


# ==========================================
# Analysis ID Generator
# ==========================================

def compute_analysis_id(
    condition_sig: str,
    w1_hash: str,
    w2_hash: str,
    algorithm: str = W3_ALGORITHM,
) -> str:
    """
    Compute deterministic analysis ID (P1-5).
    
    Args:
        condition_sig: Target condition signature
        w1_hash: Hash of W1 snapshot
        w2_hash: Hash of W2 snapshot
        algorithm: Algorithm version
        
    Returns:
        SHA256 hash (first 32 chars)
    """
    canonical = json.dumps({
        "condition_signature": condition_sig,
        "w1_snapshot_hash": w1_hash,
        "w2_snapshot_hash": w2_hash,
        "algorithm": algorithm,
    }, sort_keys=True)
    
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:32]


def compute_stats_hash(stats_dict: Dict[str, Any]) -> str:
    """
    Compute hash of stats for snapshot identification.
    
    Excludes timestamps for stability.
    """
    # Create a canonical representation excluding timestamps
    filtered = {
        k: v for k, v in stats_dict.items()
        if k not in ("updated_at", "created_at", "timestamp")
    }
    canonical = json.dumps(filtered, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("Phase 9-3 W3 Schema Test")
    print("=" * 60)
    
    # Test 1: CandidateToken
    print("\n[Test 1] CandidateToken")
    
    candidate = CandidateToken(
        token_norm="prime minister",
        s_score=0.0523,
        p_cond=0.005,
        p_global=0.0001,
        count_cond=50,
        count_global=10,
    )
    
    d = candidate.to_dict()
    restored = CandidateToken.from_dict(d)
    
    print(f"  token: {candidate.token_norm}")
    print(f"  s_score: {candidate.s_score}")
    print(f"  restored: {restored.token_norm}, {restored.s_score}")
    
    assert candidate.token_norm == restored.token_norm
    assert abs(candidate.s_score - restored.s_score) < 1e-6
    print("  ✅ PASS")
    
    # Test 2: Analysis ID (deterministic)
    print("\n[Test 2] Analysis ID (deterministic)")
    
    aid1 = compute_analysis_id("cond_abc", "w1_hash", "w2_hash", W3_ALGORITHM)
    aid2 = compute_analysis_id("cond_abc", "w1_hash", "w2_hash", W3_ALGORITHM)
    aid3 = compute_analysis_id("cond_xyz", "w1_hash", "w2_hash", W3_ALGORITHM)
    
    print(f"  aid1: {aid1}")
    print(f"  aid2: {aid2}")
    print(f"  aid3: {aid3}")
    
    assert aid1 == aid2, "Same inputs should produce same ID"
    assert aid1 != aid3, "Different inputs should produce different ID"
    print("  ✅ PASS")
    
    # Test 3: W3Record
    print("\n[Test 3] W3Record")
    
    record = W3Record(
        analysis_id=aid1,
        condition_signature="cond_abc",
        condition_factors={"source_type": "news", "language_profile": "en", "time_bucket": "2026-01"},
        w1_snapshot_hash="w1_hash",
        w2_snapshot_hash="w2_hash",
        positive_candidates=[candidate],
        negative_candidates=[],
        total_tokens_compared=1000,
        positive_count=50,
        negative_count=30,
    )
    
    json_str = record.to_json()
    restored = W3Record.from_json(json_str)
    
    print(f"  analysis_id: {record.analysis_id[:16]}...")
    print(f"  positive_candidates: {len(record.positive_candidates)}")
    print(f"  restored: {len(restored.positive_candidates)}")
    
    assert record.analysis_id == restored.analysis_id
    assert len(record.positive_candidates) == len(restored.positive_candidates)
    print("  ✅ PASS")
    
    # Test 4: Constants
    print("\n[Test 4] Constants")
    print(f"  EPSILON: {EPSILON}")
    print(f"  DEFAULT_MIN_COUNT: {DEFAULT_MIN_COUNT_FOR_W3}")
    print(f"  DEFAULT_TOP_K: {DEFAULT_TOP_K}")
    print(f"  ALGORITHM: {W3_ALGORITHM}")
    print("  ✅ PASS")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
