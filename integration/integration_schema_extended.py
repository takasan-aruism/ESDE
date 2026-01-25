"""
ESDE Phase 9-0: Integration Schema (Substrate Layer Extended)
=============================================================
Data structures for ESDE integration layer.

Changes in v5.4.7-SUB.1:
  - Added substrate_ref field to ArticleRecord (INV-AR-003)
  - Backward compatible: substrate_ref defaults to None

Spec: v5.4.7-SUB.1
"""

import uuid
import hashlib
import json
from enum import Enum
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional


# ==========================================
# Constants
# ==========================================

ENGINE_VERSION = "5.4.7-SUB.1"


# ==========================================
# Enums
# ==========================================

class DiagnosticType(Enum):
    """Diagnostic markers for observation events."""
    NONE = "none"
    EMPTY_SEGMENT = "empty_segment"
    PARSE_ERROR = "parse_error"
    LENGTH_EXCEEDED = "length_exceeded"


# ==========================================
# Environment Record
# ==========================================

@dataclass
class EnvironmentRecord:
    """
    Snapshot of processing environment.
    Used for reproducibility and debugging.
    """
    env_id: str                      # Hash of config
    synapse_version: str             # e.g., "v3.0"
    synapse_hash: str                # SHA256 of synapse file
    glossary_version: str            # e.g., "v5.3"
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "env_id": self.env_id,
            "synapse_version": self.synapse_version,
            "synapse_hash": self.synapse_hash,
            "glossary_version": self.glossary_version,
            "parameters": self.parameters,
        }


# ==========================================
# Observation Event (W0)
# ==========================================

@dataclass
class ObservationEvent:
    """
    Single observation unit from W0 (ContentGateway).
    
    IMMUTABLE after creation. Forms the foundation of all processing.
    """
    observation_id: str              # Unique ID (UUID)
    source_id: str                   # Parent article ID
    timestamp: str                   # ISO8601 observation time
    segment_index: int               # Position in article
    segment_span: Tuple[int, int]    # (start, end) in raw_text
    segment_text: Optional[str] = None  # Cache only (authoritative: raw_text + span)
    context_meta: Dict[str, Any] = field(default_factory=dict)
    observer: str = "Gateway"
    diagnostic_type: DiagnosticType = DiagnosticType.NONE
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "source_id": self.source_id,
            "timestamp": self.timestamp,
            "segment_index": self.segment_index,
            "segment_span": list(self.segment_span),
            "segment_text": self.segment_text,
            "context_meta": self.context_meta,
            "observer": self.observer,
            "diagnostic_type": self.diagnostic_type.value,
        }
    
    @staticmethod
    def generate_id() -> str:
        """Generate unique observation ID."""
        return f"obs_{uuid.uuid4().hex[:12]}"
    
    @staticmethod
    def now_iso() -> str:
        """Generate ISO8601 timestamp (UTC)."""
        return datetime.now(timezone.utc).isoformat()


# ==========================================
# Canonical Atom (Phase 8)
# ==========================================

@dataclass
class CanonicalAtom:
    """
    Single atom in a molecule.
    
    SCHEMA: INV-MOL-001 (Flat structure, no nesting)
    """
    id: str                          # Temporary reference (e.g., "aa_1")
    atom: str                        # Concept ID (e.g., "EMO.love")
    axis: Optional[str] = None       # Top-level (NOT nested in coordinates)
    level: Optional[str] = None      # Top-level (NOT nested in coordinates)
    text_ref: Optional[str] = None   # LLM-specified text reference
    span: Optional[Tuple[int, int]] = None  # System-calculated position
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "atom": self.atom,
            "axis": self.axis,
            "level": self.level,
            "text_ref": self.text_ref,
            "span": list(self.span) if self.span else None,
        }


# ==========================================
# Canonical Molecule (Phase 8)
# ==========================================

@dataclass
class CanonicalMolecule:
    """
    Molecule with tracking metadata.
    """
    molecule_id: str                           # Deterministic hash
    source_observation_ids: List[str]          # W0 references
    formula: str                               # e.g., "aa_1 × aa_2"
    active_atoms: List[CanonicalAtom]
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "molecule_id": self.molecule_id,
            "source_observation_ids": self.source_observation_ids,
            "formula": self.formula,
            "active_atoms": [a.to_dict() for a in self.active_atoms],
            "meta": self.meta,
        }
    
    def __post_init__(self):
        """Fail-fast check for legacy nested schema."""
        for atom in self.active_atoms:
            if isinstance(atom, dict) and "coordinates" in atom:
                raise ValueError(
                    "Legacy Schema Detected: 'coordinates' nesting is banned. "
                    "Use flat structure with 'axis' and 'level' at top level."
                )


# ==========================================
# Unknown Token (Phase 7)
# ==========================================

@dataclass
class UnknownToken:
    """
    Token that missed Synapse lookup.
    """
    token: str
    source_observation_id: str       # W0 reference
    segment_position: int            # Position within segment
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "token": self.token,
            "source_observation_id": self.source_observation_id,
            "segment_position": self.segment_position,
        }


# ==========================================
# Article Record (Integration Container)
# ==========================================

@dataclass
class ArticleRecord:
    """
    Integration container for a single article/context.
    
    STRUCTURE:
        - Layer 0: Raw text and segments
        - Layer 0.5: W0 observations (immutable)
        - Layer 7: Unknown tokens (weak meaning path)
        - Layer 8: Molecules (strong meaning path)
        - Emergent Slots: W1/W2/Axis (placeholders, no logic)
    
    INVARIANTS:
        INV-AR-001: source_meta is external injection only. W2 reads, never writes.
        INV-AR-002: source_meta defaults to {} for backward compatibility.
        INV-AR-003: substrate_ref is Optional; None is valid for backward compatibility.
    """
    
    # --- Meta Data ---
    article_id: str = ""             # UUIDv4 (auto-generated if empty)
    source_url: Optional[str] = None
    ingestion_time: str = ""
    environment: Optional[EnvironmentRecord] = None
    
    # --- Source Meta (for W2 condition factors) ---
    # INV-AR-001: External injection only. W2Aggregator reads, never writes/infers.
    # INV-AR-002: Default empty dict for backward compatibility with 9-0.
    # Expected keys (v1):
    #   - source_type: "news" | "dialog" | "paper" | "social" | "unknown"
    #   - language_profile: "en" | "ja" | "mixed" | "unknown"
    source_meta: Dict[str, Any] = field(default_factory=dict)
    
    # --- Substrate Layer Reference (v0.1.0+) ---
    # INV-AR-003: Optional reference to ContextRecord in Substrate Layer.
    # Phase 1: W2 continues to read source_meta directly (backward compatible).
    # Phase 2+: W2 will use ConditionSignaturePolicy to extract from traces.
    substrate_ref: Optional[str] = None  # context_id from SubstrateRegistry
    
    # --- Layer 0: Raw ---
    raw_text: str = ""               # Original full text (authoritative)
    segments: List[Tuple[int, int]] = field(default_factory=list)
    
    # --- Layer 0.5: Observation Log (W0) ---
    observations: List[ObservationEvent] = field(default_factory=list)
    
    # --- Layer 8: Structure (Strong Path) ---
    molecules: List[CanonicalMolecule] = field(default_factory=list)
    
    # --- Layer 7: Volatility (Weak Path) ---
    unknowns: List[UnknownToken] = field(default_factory=list)
    
    # --- Emergent Slots (Placeholders for Future) ---
    w1_matrix: Optional[Any] = None
    w2_matrix: Optional[Any] = None
    aruism_axis: Optional[Any] = None
    
    def __post_init__(self):
        """Initialize defaults and validate."""
        if not self.article_id:
            self.article_id = str(uuid.uuid4())
        if not self.ingestion_time:
            self.ingestion_time = datetime.now(timezone.utc).isoformat()
        
        # Fail-fast: Check molecules for legacy schema
        for mol in self.molecules:
            if isinstance(mol, CanonicalMolecule):
                continue
            if isinstance(mol, dict):
                atoms = mol.get("active_atoms", [])
                for atom in atoms:
                    if isinstance(atom, dict) and "coordinates" in atom:
                        raise ValueError(
                            "Legacy Schema Detected: 'coordinates' nesting is banned."
                        )
    
    def get_segment_text(self, index: int) -> Optional[str]:
        """Extract segment text by index."""
        if 0 <= index < len(self.segments):
            start, end = self.segments[index]
            return self.raw_text[start:end]
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "article_id": self.article_id,
            "source_url": self.source_url,
            "ingestion_time": self.ingestion_time,
            "environment": self.environment.to_dict() if self.environment else None,
            "source_meta": self.source_meta,
            "substrate_ref": self.substrate_ref,  # NEW: Substrate Layer reference
            "raw_text": self.raw_text,
            "segments": [list(s) for s in self.segments],
            "observations": [o.to_dict() for o in self.observations],
            "molecules": [m.to_dict() for m in self.molecules],
            "unknowns": [u.to_dict() for u in self.unknowns],
            "w1_matrix": self.w1_matrix,
            "w2_matrix": self.w2_matrix,
            "aruism_axis": self.aruism_axis,
        }


# ==========================================
# Factory Functions
# ==========================================

def create_observation_event(
    source_id: str,
    segment_index: int,
    segment_span: Tuple[int, int],
    segment_text: Optional[str] = None,
    context_meta: Optional[Dict[str, Any]] = None,
) -> ObservationEvent:
    """
    Factory for creating ObservationEvent.
    """
    return ObservationEvent(
        observation_id=ObservationEvent.generate_id(),
        source_id=source_id,
        timestamp=ObservationEvent.now_iso(),
        segment_index=segment_index,
        segment_span=segment_span,
        segment_text=segment_text,
        context_meta=context_meta or {},
    )


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("Phase 9-0 Schema Test (Substrate Extended)")
    print("=" * 60)
    
    # Test 1: ArticleRecord with substrate_ref
    print("\n[Test 1] ArticleRecord with substrate_ref")
    article = ArticleRecord(
        raw_text="I love you. You love me.",
        segments=[(0, 11), (12, 24)],
        source_meta={"source_type": "dialog"},
        substrate_ref="abc123def456abc123def456abc12345",
    )
    print(f"  article_id: {article.article_id[:16]}...")
    print(f"  substrate_ref: {article.substrate_ref}")
    print(f"  source_meta: {article.source_meta}")
    assert article.substrate_ref == "abc123def456abc123def456abc12345"
    print("  ✅ PASS")
    
    # Test 2: to_dict includes substrate_ref
    print("\n[Test 2] to_dict includes substrate_ref")
    d = article.to_dict()
    assert "substrate_ref" in d
    assert d["substrate_ref"] == "abc123def456abc123def456abc12345"
    print(f"  substrate_ref in dict: {d['substrate_ref']}")
    print("  ✅ PASS")
    
    # Test 3: Backward compatibility (no substrate_ref)
    print("\n[Test 3] Backward compatibility")
    old_article = ArticleRecord(
        raw_text="Test",
        source_meta={"source_type": "news"},
    )
    assert old_article.substrate_ref is None
    print(f"  substrate_ref (default): {old_article.substrate_ref}")
    print("  ✅ PASS")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
