"""
ESDE Phase 9-6: W6 Schema
=========================
W6 (Weak Structural Observation) data structures.

Theme: The Observatory - From Structure to Evidence (構造から証拠へ)

Migration Phase 3 Update:
  - policy_id: Policy identifier
  - policy_version: Policy version string
  - policy_signature_hash: SHA256 of policy configuration
  - signature_source: "policy" or "legacy"

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

Spec: v5.4.6-P9.6-Final + Migration Phase 3 v0.3.1
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timezone


# ==========================================
# Constants
# ==========================================

W6_VERSION = "v9.6.1"  # Updated for Migration Phase 3
W6_CODE_VERSION = "v5.4.8-MIG.3"

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
        data: JSON-serializable data
        length: Hash length (default: 64 = full SHA256)
        
    Returns:
        Hex string hash
    """
    canonical = get_canonical_json(data)
    return hashlib.sha256(canonical).hexdigest()[:length]


# ==========================================
# W6 Evidence Token
# ==========================================

@dataclass
class W6EvidenceToken:
    """
    Evidence token for an island.
    
    INV-W6-005: Traceable to W3Record.
    P0-X1: evidence_policy = mean_s_score_v1
    """
    token_norm: str
    evidence_score: float      # Mean S-Score across island members
    article_presence_count: int  # How many island articles contain this token
    source_w3_ids: List[str]   # W3 analysis IDs where this token appears
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_norm": self.token_norm,
            "evidence_score": round(self.evidence_score, W6_EVIDENCE_ROUNDING),
            "article_presence_count": self.article_presence_count,
            "source_w3_ids": sorted(self.source_w3_ids),  # INV-W6-006
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
# W6 Representative Article
# ==========================================

@dataclass
class W6RepresentativeArticle:
    """
    Representative article for an island.
    
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
    # Identity
    island_id: str
    size: int
    cohesion_score: float
    
    # Digest (from W5 representative_vector)
    # List of (condition_sig, value) sorted by |value| desc
    representative_vec_digest: List[Tuple[str, float]]
    
    # Evidence (from W3 via W4)
    evidence_tokens: List[W6EvidenceToken]
    used_w3_digest: str  # Hash of W3 analysis IDs used
    
    # Representative Articles (from W0)
    representative_articles: List[W6RepresentativeArticle]
    
    # Members (for scope verification)
    member_ids: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "island_id": self.island_id,
            "size": self.size,
            "cohesion_score": round(self.cohesion_score, 8),
            "representative_vec_digest": [
                [sig, round(val, 8)] for sig, val in self.representative_vec_digest
            ],
            "evidence_tokens": [t.to_dict() for t in self.evidence_tokens],
            "used_w3_digest": self.used_w3_digest,
            "representative_articles": [a.to_dict() for a in self.representative_articles],
            "member_ids": sorted(self.member_ids),  # INV-W6-006
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'W6IslandDetail':
        return cls(
            island_id=data["island_id"],
            size=data["size"],
            cohesion_score=data["cohesion_score"],
            representative_vec_digest=[
                (item[0], item[1]) for item in data.get("representative_vec_digest", [])
            ],
            evidence_tokens=[
                W6EvidenceToken.from_dict(t) for t in data.get("evidence_tokens", [])
            ],
            used_w3_digest=data.get("used_w3_digest", ""),
            representative_articles=[
                W6RepresentativeArticle.from_dict(a) for a in data.get("representative_articles", [])
            ],
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
    
    Migration Phase 3 Update:
      - policy_id: Policy identifier (or None for Legacy mode)
      - policy_version: Policy version string
      - policy_signature_hash: SHA256 of policy configuration
      - signature_source: "policy" or "legacy"
    
    INV-W6-002: Deterministic given same inputs.
    INV-W6-009: Input sets must match W5 member scope.
    """
    # === Identity ===
    observation_id: str           # SHA256 of canonical content
    input_structure_id: str       # W5 structure ID
    analysis_scope_id: str        # Scope identifier
    
    # === Islands ===
    islands: List[W6IslandDetail]
    
    # === Topology ===
    topology_pairs: List[W6TopologyPair]
    
    # === Version Tracking (INV-W6-008) ===
    tokenizer_version: str
    normalizer_version: str
    w3_versions_digest: str       # Hash of W3 versions used
    code_version: str = W6_CODE_VERSION
    
    # === Migration Phase 3: Policy Metadata ===
    policy_id: Optional[str] = None
    policy_version: Optional[str] = None
    policy_signature_hash: Optional[str] = None
    signature_source: str = "legacy"  # "policy" or "legacy"
    
    # === Parameters ===
    params: Dict[str, Any] = field(default_factory=dict)
    
    # === Noise (unassigned articles) ===
    noise_count: int = 0
    noise_ids: List[str] = field(default_factory=list)
    
    # === Timestamp (non-canonical) ===
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "observation_id": self.observation_id,
            "input_structure_id": self.input_structure_id,
            "analysis_scope_id": self.analysis_scope_id,
            "islands": [i.to_dict() for i in self.islands],
            "topology_pairs": [p.to_dict() for p in self.topology_pairs],
            "tokenizer_version": self.tokenizer_version,
            "normalizer_version": self.normalizer_version,
            "w3_versions_digest": self.w3_versions_digest,
            "code_version": self.code_version,
            # Migration Phase 3: Policy metadata
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_signature_hash": self.policy_signature_hash,
            "signature_source": self.signature_source,
            # Other fields
            "params": self.params,
            "noise_count": self.noise_count,
            "noise_ids": sorted(self.noise_ids),  # INV-W6-006
            "created_at": self.created_at,
        }
        return result
    
    def to_canonical_dict(self) -> Dict[str, Any]:
        """
        Convert to canonical dictionary for hashing.
        
        Excludes: created_at (non-deterministic)
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
            # Migration Phase 3: Policy metadata
            policy_id=data.get("policy_id"),
            policy_version=data.get("policy_version"),
            policy_signature_hash=data.get("policy_signature_hash"),
            signature_source=data.get("signature_source", "legacy"),
            # Other fields
            params=data.get("params", {}),
            noise_count=data.get("noise_count", 0),
            noise_ids=data.get("noise_ids", []),
            created_at=data.get("created_at", ""),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'W6Observatory':
        return cls.from_dict(json.loads(json_str))
    
    def get_policy_metadata(self) -> Dict[str, Any]:
        """
        Get policy metadata dict (Migration Phase 3).
        
        Returns:
            Dict with policy_id, policy_version, policy_signature_hash, signature_source
        """
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_signature_hash": self.policy_signature_hash,
            "signature_source": self.signature_source,
            "analysis_scope_id": self.analysis_scope_id,
        }


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
    """
    d1 = obs1.to_canonical_dict()
    d2 = obs2.to_canonical_dict()
    
    # Compare hashes
    h1 = compute_canonical_hash(d1)
    h2 = compute_canonical_hash(d2)
    
    differences = []
    
    if h1 != h2:
        # Find differences
        for key in set(d1.keys()) | set(d2.keys()):
            v1 = d1.get(key)
            v2 = d2.get(key)
            if v1 != v2:
                differences.append({
                    "field": key,
                    "obs1": str(v1)[:100] if v1 else None,
                    "obs2": str(v2)[:100] if v2 else None,
                })
    
    return {
        "match": h1 == h2,
        "hash1": h1,
        "hash2": h2,
        "differences": differences[:10],  # Limit
    }


# ==========================================
# Factory Function for Migration Phase 3
# ==========================================

def create_w6_observatory_with_policy(
    observation_id: str,
    input_structure_id: str,
    islands: List[W6IslandDetail],
    topology_pairs: List[W6TopologyPair],
    tokenizer_version: str,
    normalizer_version: str,
    w3_versions_digest: str,
    execution_context=None,  # Optional ExecutionContext from runner
    params: Dict[str, Any] = None,
    noise_count: int = 0,
    noise_ids: List[str] = None,
) -> W6Observatory:
    """
    Factory function to create W6Observatory with policy metadata.
    
    Migration Phase 3: Automatically extracts policy metadata from ExecutionContext.
    
    Args:
        observation_id: Observation identifier
        input_structure_id: W5 structure ID
        islands: List of W6IslandDetail
        topology_pairs: List of W6TopologyPair
        tokenizer_version: Tokenizer version
        normalizer_version: Normalizer version
        w3_versions_digest: Hash of W3 versions
        execution_context: Optional ExecutionContext from StatisticsPipelineRunner
        params: Optional parameters dict
        noise_count: Number of noise articles
        noise_ids: List of noise article IDs
        
    Returns:
        W6Observatory with policy metadata populated
    """
    # Extract policy metadata from ExecutionContext
    policy_id = None
    policy_version = None
    policy_signature_hash = None
    signature_source = "legacy"
    analysis_scope_id = ""
    
    if execution_context is not None:
        metadata = execution_context.to_metadata_dict()
        policy_id = metadata.get("policy_id")
        policy_version = metadata.get("policy_version")
        policy_signature_hash = metadata.get("policy_signature_hash")
        signature_source = metadata.get("signature_source", "legacy")
        analysis_scope_id = metadata.get("analysis_scope_id", "")
    
    return W6Observatory(
        observation_id=observation_id,
        input_structure_id=input_structure_id,
        analysis_scope_id=analysis_scope_id,
        islands=islands,
        topology_pairs=topology_pairs,
        tokenizer_version=tokenizer_version,
        normalizer_version=normalizer_version,
        w3_versions_digest=w3_versions_digest,
        code_version=W6_CODE_VERSION,
        policy_id=policy_id,
        policy_version=policy_version,
        policy_signature_hash=policy_signature_hash,
        signature_source=signature_source,
        params=params or {},
        noise_count=noise_count,
        noise_ids=noise_ids or [],
    )
