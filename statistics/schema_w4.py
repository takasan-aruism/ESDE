"""
ESDE Phase 9-4: W4 Schema
=========================
W4 (Weak Structural Projection) data structures.

Theme: The Resonance of Weakness (弱さの共鳴・構造射影)

Design Principles:
  - W4 = ArticleRecord projected onto W3 axis candidates
  - Produces "resonance vector" (per-condition specificity scores)
  - No labeling, no interpretation (INV-W4-001)

Mathematical Model:
  R(A, C) = Σ count(t, A) × S(t, C)
  
  Where:
    - count(t, A) = token count in article A
    - S(t, C) = S-Score from W3 for token t under condition C

Invariants:
  INV-W4-001 (No Labeling):
      Output keys must be condition_signature (hash).
      No natural language labels like "News" allowed.
  
  INV-W4-002 (Deterministic):
      Same article + same W3 set produces identical scores.
      (floating-point tolerance acceptable)
  
  INV-W4-003 (Recomputable):
      W4 can always be regenerated from W0 (ArticleRecord) + W3.
  
  INV-W4-004 (Full S-Score Usage):
      W4 uses both positive AND negative candidates from W3.
      (token_norm -> s_score mapping includes all signs)
  
  INV-W4-005 (Immutable Input):
      W4Projector does NOT modify ArticleRecord or W3Record.
  
  INV-W4-006 (Tokenization Canon):
      W4 MUST use W1Tokenizer + normalize_token from statistics package.
      No independent tokenization implementation allowed.

Spec: v5.4.4-P9.4
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import json
import hashlib
import uuid


# ==========================================
# Constants
# ==========================================

W4_VERSION = "v9.4.0"
W4_ALGORITHM = "DotProduct-v1"
W4_PROJECTION_NORM = "raw"  # v9.4: no length/variance normalization

# Output directory
DEFAULT_W4_OUTPUT_DIR = "data/stats/w4_projections"


# ==========================================
# W4 Record
# ==========================================

@dataclass
class W4Record:
    """
    W4: Resonance vector for a single article.
    
    Maps an ArticleRecord to a vector space defined by W3 axis candidates.
    Each dimension corresponds to a condition_signature.
    
    INV-W4-001: Keys are condition_signature only, no labels.
    INV-W4-002: Deterministic given same inputs.
    INV-W4-003: Recomputable from W0 + W3.
    """
    
    # --- Identity ---
    article_id: str                         # Link to ArticleRecord
    w4_analysis_id: str                     # Deterministic ID (P0-1)
    w4_run_id: str = ""                     # UUID for ops/logging
    computed_at: str = ""                   # ISO8601 UTC (P1-1 rename from timestamp)
    
    # --- The Resonance Vector ---
    # Key: condition_signature (from W3)
    # Value: resonance_score (dot product)
    # Example: {"d183...": 15.4, "a9f2...": -2.1}
    resonance_vector: Dict[str, float] = field(default_factory=dict)
    
    # --- Traceability (P0-2: map instead of list) ---
    # Key: condition_signature
    # Value: w3_analysis_id used for that condition
    used_w3: Dict[str, str] = field(default_factory=dict)
    
    # --- Meta (P0-3: length bias awareness) ---
    token_count: int = 0                    # Total valid tokens in article
    tokenizer_version: str = ""             # e.g., "hybrid_v1"
    normalizer_version: str = ""            # e.g., "v9.1.0"
    projection_norm: str = W4_PROJECTION_NORM  # v9.4: "raw" (未正規化)
    algorithm: str = W4_ALGORITHM           # "DotProduct-v1"
    
    def __post_init__(self):
        if not self.computed_at:
            self.computed_at = datetime.now(timezone.utc).isoformat()
        if not self.w4_run_id:
            self.w4_run_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "article_id": self.article_id,
            "w4_analysis_id": self.w4_analysis_id,
            "w4_run_id": self.w4_run_id,
            "computed_at": self.computed_at,
            "resonance_vector": {k: round(v, 8) for k, v in sorted(self.resonance_vector.items())},
            "used_w3": dict(sorted(self.used_w3.items())),
            "token_count": self.token_count,
            "tokenizer_version": self.tokenizer_version,
            "normalizer_version": self.normalizer_version,
            "projection_norm": self.projection_norm,
            "algorithm": self.algorithm,
        }
    
    def to_canonical_dict(self) -> Dict[str, Any]:
        """
        Canonical dict for determinism verification.
        
        Excludes:
          - w4_run_id (operational, non-deterministic)
          - computed_at (changes per run)
        """
        return {
            "article_id": self.article_id,
            "w4_analysis_id": self.w4_analysis_id,
            "resonance_vector": {k: round(v, 8) for k, v in sorted(self.resonance_vector.items())},
            "used_w3": dict(sorted(self.used_w3.items())),
            "token_count": self.token_count,
            "tokenizer_version": self.tokenizer_version,
            "normalizer_version": self.normalizer_version,
            "projection_norm": self.projection_norm,
            "algorithm": self.algorithm,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, sort_keys=True)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'W4Record':
        """Create from dict."""
        return cls(
            article_id=data["article_id"],
            w4_analysis_id=data["w4_analysis_id"],
            w4_run_id=data.get("w4_run_id", ""),
            computed_at=data.get("computed_at", ""),
            resonance_vector=data.get("resonance_vector", {}),
            used_w3=data.get("used_w3", {}),
            token_count=data.get("token_count", 0),
            tokenizer_version=data.get("tokenizer_version", ""),
            normalizer_version=data.get("normalizer_version", ""),
            projection_norm=data.get("projection_norm", W4_PROJECTION_NORM),
            algorithm=data.get("algorithm", W4_ALGORITHM),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'W4Record':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


# ==========================================
# Analysis ID Generator (P0-1)
# ==========================================

def compute_w4_analysis_id(
    article_id: str,
    used_w3: Dict[str, str],
    tokenizer_version: str,
    normalizer_version: str,
    algorithm: str = W4_ALGORITHM,
) -> str:
    """
    Compute deterministic W4 analysis ID.
    
    P0-1 Compliance:
      - Includes article_id
      - Includes sorted w3_analysis_ids
      - Includes version info for reproducibility
    
    Args:
        article_id: ArticleRecord.article_id
        used_w3: {condition_signature: w3_analysis_id}
        tokenizer_version: Tokenizer version string
        normalizer_version: Normalizer version string
        algorithm: Algorithm identifier
        
    Returns:
        SHA256 hash (first 32 chars)
    """
    # Sort w3_analysis_ids for determinism
    sorted_w3_ids = [used_w3[k] for k in sorted(used_w3.keys())]
    
    canonical = json.dumps({
        "article_id": article_id,
        "w3_analysis_ids": sorted_w3_ids,
        "tokenizer_version": tokenizer_version,
        "normalizer_version": normalizer_version,
        "algorithm": algorithm,
    }, sort_keys=True)
    
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:32]


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("Phase 9-4 W4 Schema Test")
    print("=" * 60)
    
    # Test 1: W4Record creation
    print("\n[Test 1] W4Record creation")
    
    # Use hash-like signatures (as in real operation)
    cond_sig_1 = "d183f5a4e2b7c9d1a3f6e8b2c4d7f9a1"  # news-like
    cond_sig_2 = "a9f2b3c4d5e6f7a8b9c0d1e2f3a4b5c6"  # dialog-like
    
    used_w3 = {
        cond_sig_1: "w3_analysis_001",
        cond_sig_2: "w3_analysis_002",
    }
    
    analysis_id = compute_w4_analysis_id(
        article_id="article_001",
        used_w3=used_w3,
        tokenizer_version="hybrid_v1",
        normalizer_version="v9.1.0",
    )
    
    record = W4Record(
        article_id="article_001",
        w4_analysis_id=analysis_id,
        resonance_vector={
            cond_sig_1: 15.4,
            cond_sig_2: -2.1,
        },
        used_w3=used_w3,
        token_count=150,
        tokenizer_version="hybrid_v1",
        normalizer_version="v9.1.0",
    )
    
    print(f"  article_id: {record.article_id}")
    print(f"  w4_analysis_id: {record.w4_analysis_id}")
    print(f"  resonance_vector: {record.resonance_vector}")
    print(f"  token_count: {record.token_count}")
    print("  ✅ PASS")
    
    # Test 2: Deterministic analysis ID (P0-1)
    print("\n[Test 2] Deterministic analysis ID")
    
    aid1 = compute_w4_analysis_id("art1", {"c1": "w3_1", "c2": "w3_2"}, "tok_v1", "norm_v1")
    aid2 = compute_w4_analysis_id("art1", {"c2": "w3_2", "c1": "w3_1"}, "tok_v1", "norm_v1")  # Different order
    aid3 = compute_w4_analysis_id("art2", {"c1": "w3_1", "c2": "w3_2"}, "tok_v1", "norm_v1")  # Different article
    
    print(f"  aid1: {aid1}")
    print(f"  aid2: {aid2}")
    print(f"  aid3: {aid3}")
    
    assert aid1 == aid2, "Order should not affect ID"
    assert aid1 != aid3, "Different article should produce different ID"
    print("  ✅ PASS")
    
    # Test 3: JSON serialization
    print("\n[Test 3] JSON serialization")
    
    json_str = record.to_json()
    restored = W4Record.from_json(json_str)
    
    print(f"  JSON length: {len(json_str)} chars")
    print(f"  Restored article_id: {restored.article_id}")
    print(f"  Restored analysis_id: {restored.w4_analysis_id}")
    
    assert record.article_id == restored.article_id
    assert record.w4_analysis_id == restored.w4_analysis_id
    assert abs(record.resonance_vector[cond_sig_1] - restored.resonance_vector[cond_sig_1]) < 1e-6
    print("  ✅ PASS")
    
    # Test 4: Canonical dict (excludes operational fields)
    print("\n[Test 4] Canonical dict (determinism verification)")
    
    canonical = record.to_canonical_dict()
    
    print(f"  Keys: {list(canonical.keys())}")
    
    assert "w4_run_id" not in canonical, "w4_run_id should be excluded"
    assert "computed_at" not in canonical, "computed_at should be excluded"
    assert "w4_analysis_id" in canonical, "w4_analysis_id should be included"
    print("  ✅ PASS")
    
    # Test 5: INV-W4-001 compliance (No labeling)
    print("\n[Test 5] INV-W4-001: No labeling check")
    
    # resonance_vector keys should be hashes, not labels
    for key in record.resonance_vector.keys():
        assert not any(label in key.lower() for label in ["news", "dialog", "paper", "social"]), \
            f"Key contains label: {key}"
    
    print("  Keys are condition signatures (no natural language labels)")
    print("  ✅ PASS")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("\nPhase 9-4 W4 Schema is ready.")
