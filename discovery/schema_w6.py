"""
ESDE Phase 9-6: W6 Schema
=========================
W6 (Weak Structural Observation) data structures.

Theme: The Observatory - From Structure to Evidence (構造から証拠へ)

Design Philosophy:
  - W6 = Observation Window, NOT computation layer
  - No New Math: Transforms/extracts only, no new statistics
  - Strict Determinism: Bit-level reproducibility guaranteed
  - Evidence-based: All outputs traceable to W3/W4/W5

Invariants:
  INV-W6-001 (No Synthetic Labels):
      No natural language categories, no LLM summaries.
      Evidence tokens are W3-derived only.
      
  INV-W6-002 (Deterministic Export):
      Same input (W5+Params) produces bit-identical output.
      
  INV-W6-003 (Read Only):
      W1-W5 data is never modified.
      
  INV-W6-004 (No New Math):
      No new statistical models. Topology = W5 vector distance only.
      
  INV-W6-005 (Evidence Provenance):
      All evidence tokens/articles traceable to W3/W0.
      
  INV-W6-006 (Stable Ordering):
      All list outputs have complete tie-break rules.
      
  INV-W6-007 (No Hypothesis):
      No "Axis Hypothesis", "Confidence" or judgment logic.
      
  INV-W6-008 (Strict Versioning):
      Tokenizer, Normalizer, W3 version compatibility tracked.
      
  INV-W6-009 (Scope Closure):
      W5 member set must match W4/ArticleRecord input sets exactly.

Spec: v5.4.6-P9.6-Final
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timezone


# ==========================================
# Constants
# ==========================================

W6_VERSION = "v9.6.0"
W6_CODE_VERSION = "v5.4.6-P9.6"

# Fixed Policies (P0-2, P1-2, P1-3)
W6_EVIDENCE_POLICY = "mean_s_score_v1"
W6_SNIPPET_POLICY = "head_chars_v1"
W6_METRIC_POLICY = "cosine_dist_v1"
W6_DIGEST_POLICY = "abs_val_desc_v1"

# Policy Parameters
W6_SNIPPET_LENGTH = 200
W6_DIGEST_K = 10
W6_EVIDENCE_K = 20
W6_REP_ARTICLE_K = 3
W6_TOPOLOGY_K = 10  # Top-N pairs to include (P1-b)

# Rounding
W6_DISTANCE_ROUNDING = 12
W6_EVIDENCE_ROUNDING = 8

# Output directory
DEFAULT_W6_OUTPUT_DIR = "data/discovery/w6_observations"


# ==========================================
# Canonical JSON Utility (inherited from W5)
# ==========================================

def get_canonical_json(data: Any) -> bytes:
    """
    Safe canonical JSON serialization for hashing.
    
    INV-W6-002: All ID generation must use this function.
    String concatenation (join) is strictly forbidden.
    
    Args:
        data: Any JSON-serializable data
        
    Returns:
        UTF-8 encoded canonical JSON bytes
    """
    return json.dumps(
        data,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True
    ).encode('utf-8')


def compute_canonical_hash(data: Any, length: int = 64) -> str:
    """
    Compute SHA256 hash of canonical JSON.
    
    Args:
        data: Data to hash
        length: Hash string length (default: full 64 chars)
        
    Returns:
        Hex digest string
    """
    return hashlib.sha256(get_canonical_json(data)).hexdigest()[:length]


# ==========================================
# Evidence Token
# ==========================================

@dataclass
class W6EvidenceToken:
    """
    A token identified as evidence for an island.
    
    INV-W6-005: Must be traceable to W3.
    
    P0-X1 Formula (mean_s_score_v1):
      evidence_score = mean_r( S(token, cond(r)) * I[token in article(r)] )
      where:
        - r iterates over all articles in the island
        - cond(r) = the condition_signature used by article r
        - I[...] = 1 if token exists in article, else 0
        - mean denominator = total island article count (including zeros)
    """
    token_norm: str
    evidence_score: float      # Rounded to W6_EVIDENCE_ROUNDING
    article_presence_count: int  # How many island articles contain this token
    source_w3_ids: List[str]   # W3 analysis_ids where this token appears
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_norm": self.token_norm,
            "evidence_score": round(self.evidence_score, W6_EVIDENCE_ROUNDING),
            "article_presence_count": self.article_presence_count,
            "source_w3_ids": sorted(self.source_w3_ids),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'W6EvidenceToken':
        return cls(
            token_norm=data["token_norm"],
            evidence_score=data["evidence_score"],
            article_presence_count=data.get("article_presence_count", 0),
            source_w3_ids=data.get("source_w3_ids", []),
        )


# ==========================================
# Representative Article
# ==========================================

@dataclass
class W6RepresentativeArticle:
    """
    A sample article from an island with snippet.
    
    INV-W6-005: Traceable to W0 (ArticleRecord).
    P1-2: snippet_policy = head_chars_v1 (first 200 chars)
    """
    article_id: str
    w4_analysis_id: str        # Traceability to W4
    resonance_magnitude: float # |resonance_vector| for sorting
    snippet: str               # First 200 chars (policy: head_chars_v1)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "article_id": self.article_id,
            "w4_analysis_id": self.w4_analysis_id,
            "resonance_magnitude": round(self.resonance_magnitude, 8),
            "snippet": self.snippet,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'W6RepresentativeArticle':
        return cls(
            article_id=data["article_id"],
            w4_analysis_id=data["w4_analysis_id"],
            resonance_magnitude=data.get("resonance_magnitude", 0.0),
            snippet=data.get("snippet", ""),
        )


# ==========================================
# W6 Island Detail
# ==========================================

@dataclass
class W6IslandDetail:
    """
    Detailed observation of a W5 island.
    
    Contains evidence tokens, representative articles, and provenance.
    """
    # Identity (from W5)
    island_id: str
    size: int
    cohesion_score: float
    
    # Digest (P1-1: abs_val_desc_v1)
    # Sorted by (-abs(value), cond_sig asc), Top-K=10
    representative_vec_digest: List[Tuple[str, float]]
    
    # Evidence Tokens (P0-X1: mean_s_score_v1)
    # Sorted by (-evidence_score_rounded, token_norm asc)
    evidence_tokens: List[W6EvidenceToken]
    
    # Provenance Digest (P0-3)
    # SHA256(sorted([used_w3 values for all members]))
    used_w3_digest: str
    
    # Representative Articles (P1-2: head_chars_v1)
    # Sorted by (-resonance_magnitude, article_id asc)
    representative_articles: List[W6RepresentativeArticle]
    
    # Member IDs (for reference)
    member_ids: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "island_id": self.island_id,
            "size": self.size,
            "cohesion_score": round(self.cohesion_score, 6),
            "representative_vec_digest": [
                [k, round(v, 8)] for k, v in self.representative_vec_digest
            ],
            "evidence_tokens": [t.to_dict() for t in self.evidence_tokens],
            "used_w3_digest": self.used_w3_digest,
            "representative_articles": [a.to_dict() for a in self.representative_articles],
            "member_ids": self.member_ids,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'W6IslandDetail':
        return cls(
            island_id=data["island_id"],
            size=data["size"],
            cohesion_score=data["cohesion_score"],
            representative_vec_digest=[tuple(x) for x in data.get("representative_vec_digest", [])],
            evidence_tokens=[W6EvidenceToken.from_dict(t) for t in data.get("evidence_tokens", [])],
            used_w3_digest=data.get("used_w3_digest", ""),
            representative_articles=[W6RepresentativeArticle.from_dict(a) for a in data.get("representative_articles", [])],
            member_ids=data.get("member_ids", []),
        )


# ==========================================
# W6 Topology Pair
# ==========================================

@dataclass
class W6TopologyPair:
    """
    Distance between two islands.
    
    INV-W6-004: Distance computed from W5 representative_vector ONLY.
    P1-3: metric_policy = cosine_dist_v1 (1.0 - round(cos_sim, 12))
    """
    # Identity (P1-a: canonical JSON hash)
    pair_id: str
    
    # Islands
    island_a_id: str
    island_b_id: str
    
    # Distance
    distance: float            # 1.0 - cos_sim, rounded
    metric: str                # "cosine_dist_v1"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "island_a_id": self.island_a_id,
            "island_b_id": self.island_b_id,
            "distance": round(self.distance, W6_DISTANCE_ROUNDING),
            "metric": self.metric,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'W6TopologyPair':
        return cls(
            pair_id=data["pair_id"],
            island_a_id=data["island_a_id"],
            island_b_id=data["island_b_id"],
            distance=data["distance"],
            metric=data["metric"],
        )


# ==========================================
# W6 Observatory (Main Output)
# ==========================================

@dataclass
class W6Observatory:
    """
    W6 Observatory: Complete observation of a W5 structure.
    
    The main output of Phase 9-6.
    
    INV-W6-002: Deterministic given same inputs.
    INV-W6-009: Input sets must match W5 member scope.
    """
    # Identity (P0-4: Strict Canonical Hash)
    observation_id: str
    
    # Input Reference
    input_structure_id: str    # W5 Structure ID
    
    # Scope (P0-X2)
    analysis_scope_id: str     # e.g., "news_202501"
    
    # Observed Islands
    islands: List[W6IslandDetail]
    
    # Topology (P0-1: W5 vector only)
    topology_pairs: List[W6TopologyPair]
    
    # Version Tracking (P0-3, INV-W6-008)
    tokenizer_version: str
    normalizer_version: str
    w3_versions_digest: str    # Hash of input W3 versions
    code_version: str          # e.g., "v5.4.6-P9.6"
    
    # Parameters (Fixed Policies)
    params: Dict[str, Any]
    
    # Noise Summary
    noise_count: int
    noise_ids: List[str]
    
    # Non-deterministic (INV-W6-002: excluded from canonical)
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "input_structure_id": self.input_structure_id,
            "analysis_scope_id": self.analysis_scope_id,
            "islands": [i.to_dict() for i in self.islands],
            "topology_pairs": [p.to_dict() for p in self.topology_pairs],
            "tokenizer_version": self.tokenizer_version,
            "normalizer_version": self.normalizer_version,
            "w3_versions_digest": self.w3_versions_digest,
            "code_version": self.code_version,
            "params": self.params,
            "noise_count": self.noise_count,
            "noise_ids": self.noise_ids,
            "created_at": self.created_at,
        }
    
    def get_canonical_dict(self) -> Dict[str, Any]:
        """
        Returns dict for determinism comparison.
        
        Excludes: created_at
        """
        d = self.to_dict()
        d.pop('created_at', None)
        return d
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, sort_keys=True)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'W6Observatory':
        return cls(
            observation_id=data["observation_id"],
            input_structure_id=data["input_structure_id"],
            analysis_scope_id=data.get("analysis_scope_id", ""),
            islands=[W6IslandDetail.from_dict(i) for i in data.get("islands", [])],
            topology_pairs=[W6TopologyPair.from_dict(p) for p in data.get("topology_pairs", [])],
            tokenizer_version=data.get("tokenizer_version", ""),
            normalizer_version=data.get("normalizer_version", ""),
            w3_versions_digest=data.get("w3_versions_digest", ""),
            code_version=data.get("code_version", W6_CODE_VERSION),
            params=data.get("params", {}),
            noise_count=data.get("noise_count", 0),
            noise_ids=data.get("noise_ids", []),
            created_at=data.get("created_at", ""),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'W6Observatory':
        return cls.from_dict(json.loads(json_str))


# ==========================================
# Comparison Utility
# ==========================================

def compare_w6_observations(
    obs1: W6Observatory,
    obs2: W6Observatory,
) -> Dict[str, Any]:
    """
    Compare two W6Observations for determinism verification.
    
    INV-W6-002: Uses canonical dict (excludes created_at).
    
    Args:
        obs1: First W6Observatory
        obs2: Second W6Observatory
        
    Returns:
        Comparison result dict
    """
    diffs = []
    
    c1 = obs1.get_canonical_dict()
    c2 = obs2.get_canonical_dict()
    
    # Compare observation_id
    if c1["observation_id"] != c2["observation_id"]:
        diffs.append(f"observation_id: {c1['observation_id'][:16]}... vs {c2['observation_id'][:16]}...")
    
    # Compare island count
    if len(c1.get("islands", [])) != len(c2.get("islands", [])):
        diffs.append(f"island_count: {len(c1.get('islands', []))} vs {len(c2.get('islands', []))}")
    
    # Compare topology pair count
    if len(c1.get("topology_pairs", [])) != len(c2.get("topology_pairs", [])):
        diffs.append(f"topology_count: {len(c1.get('topology_pairs', []))} vs {len(c2.get('topology_pairs', []))}")
    
    # Compare noise
    if c1.get("noise_count") != c2.get("noise_count"):
        diffs.append(f"noise_count: {c1.get('noise_count')} vs {c2.get('noise_count')}")
    
    return {
        "match": len(diffs) == 0,
        "differences": diffs[:10],
        "total_differences": len(diffs),
    }


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("Phase 9-6 W6 Schema Test")
    print("=" * 60)
    
    # Test 1: W6EvidenceToken
    print("\n[Test 1] W6EvidenceToken")
    
    ev_token = W6EvidenceToken(
        token_norm="prime minister",
        evidence_score=0.0523456789,
        article_presence_count=5,
        source_w3_ids=["w3_001", "w3_002"],
    )
    
    d = ev_token.to_dict()
    restored = W6EvidenceToken.from_dict(d)
    
    print(f"  token: {ev_token.token_norm}")
    print(f"  score (rounded): {d['evidence_score']}")
    print(f"  presence: {ev_token.article_presence_count}")
    
    assert restored.token_norm == ev_token.token_norm
    print("  ✅ PASS")
    
    # Test 2: W6TopologyPair
    print("\n[Test 2] W6TopologyPair")
    
    pair_id = compute_canonical_hash({
        "island_a": "island_001",
        "island_b": "island_002",
        "metric": W6_METRIC_POLICY,
    })
    
    pair = W6TopologyPair(
        pair_id=pair_id,
        island_a_id="island_001",
        island_b_id="island_002",
        distance=0.35,
        metric=W6_METRIC_POLICY,
    )
    
    print(f"  pair_id: {pair.pair_id[:16]}...")
    print(f"  distance: {pair.distance}")
    print("  ✅ PASS")
    
    # Test 3: W6IslandDetail
    print("\n[Test 3] W6IslandDetail")
    
    island_detail = W6IslandDetail(
        island_id="island_001",
        size=5,
        cohesion_score=0.85,
        representative_vec_digest=[("cond_a", 0.5), ("cond_b", -0.3)],
        evidence_tokens=[ev_token],
        used_w3_digest="abc123",
        representative_articles=[],
        member_ids=["art_001", "art_002"],
    )
    
    d = island_detail.to_dict()
    print(f"  island_id: {island_detail.island_id}")
    print(f"  evidence_tokens: {len(island_detail.evidence_tokens)}")
    print("  ✅ PASS")
    
    # Test 4: W6Observatory
    print("\n[Test 4] W6Observatory")
    
    obs = W6Observatory(
        observation_id="obs_001",
        input_structure_id="struct_001",
        analysis_scope_id="test_scope",
        islands=[island_detail],
        topology_pairs=[pair],
        tokenizer_version="hybrid_v1",
        normalizer_version="v9.1.0",
        w3_versions_digest="w3_digest",
        code_version=W6_CODE_VERSION,
        params={
            "evidence_policy": W6_EVIDENCE_POLICY,
            "snippet_policy": W6_SNIPPET_POLICY,
        },
        noise_count=2,
        noise_ids=["noise_001", "noise_002"],
    )
    
    print(f"  observation_id: {obs.observation_id}")
    print(f"  islands: {len(obs.islands)}")
    print(f"  topology_pairs: {len(obs.topology_pairs)}")
    
    # Test canonical dict excludes created_at
    canonical = obs.get_canonical_dict()
    assert "created_at" not in canonical
    print("  canonical dict excludes created_at ✓")
    
    print("  ✅ PASS")
    
    # Test 5: JSON round-trip
    print("\n[Test 5] JSON serialization")
    
    json_str = obs.to_json()
    restored = W6Observatory.from_json(json_str)
    
    assert restored.observation_id == obs.observation_id
    assert len(restored.islands) == len(obs.islands)
    print(f"  JSON length: {len(json_str)} chars")
    print("  ✅ PASS")
    
    print("\n" + "=" * 60)
    print("All schema tests passed! ✅")
