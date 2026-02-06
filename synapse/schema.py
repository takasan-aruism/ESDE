"""
ESDE Synapse - Patch Schema
============================
Data model for Synapse overlay patches.

Design Spec v2.1 §1: Edge Identity
  - edge_key = "{synset_id}::{atom_id}" (unique identifier)
  - op = "add_edge" | "disable_edge"

3AI Approval: Gemini (design) → GPT (audit) → Claude (implementation)
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional


@dataclass
class SynapsePatchEntry:
    """
    A single patch operation on Synapse.
    
    Design Spec v2.1 §1:
      edge_key uniquely identifies an edge as {synset_id}::{atom_id}.
      op determines whether to add or disable.
    
    Attributes:
        op: "add_edge" or "disable_edge"
        edge_key: "{synset_id}::{atom_id}" — canonical identifier
        synset_id: WordNet synset ID (e.g., "kill.v.01")
        atom: ESDE Atom ID (e.g., "EXS.death")
        score: Similarity score (for add_edge)
        reason: Origin identifier (e.g., "auto_proposal_v2.0", "manual_review")
        metadata: Additional trace info (source patch version, rewrite_pack_id, etc.)
    """
    op: str  # "add_edge" | "disable_edge"
    edge_key: str  # "{synset_id}::{atom_id}"
    synset_id: str
    atom: str
    score: float = 0.0
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate invariants."""
        if self.op not in ("add_edge", "disable_edge"):
            raise ValueError(f"Invalid op: {self.op!r}. Must be 'add_edge' or 'disable_edge'.")
        
        expected_key = f"{self.synset_id}::{self.atom}"
        if self.edge_key != expected_key:
            raise ValueError(
                f"edge_key mismatch: got {self.edge_key!r}, "
                f"expected {expected_key!r} from synset_id={self.synset_id!r}, atom={self.atom!r}"
            )
    
    @classmethod
    def make_key(cls, synset_id: str, atom: str) -> str:
        """Generate canonical edge_key."""
        return f"{synset_id}::{atom}"
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SynapsePatchEntry":
        """Construct from dict (JSON deserialization)."""
        return cls(
            op=d["op"],
            edge_key=d["edge_key"],
            synset_id=d["synset_id"],
            atom=d["atom"],
            score=d.get("score", 0.0),
            reason=d.get("reason", ""),
            metadata=d.get("metadata", {}),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)
    
    def to_synapse_edge(self) -> Dict[str, Any]:
        """
        Convert to Synapse base format edge dict.
        
        The base Synapse JSON stores edges as:
          {"concept_id": "EXS.death", "raw_score": 0.85, ...}
        
        This produces a compatible edge with patch metadata.
        """
        return {
            "concept_id": self.atom,
            "raw_score": self.score,
            "patch_source": self.reason,
            "patch_metadata": self.metadata,
        }
