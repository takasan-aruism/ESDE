"""
ESDE Phase 9-0: Integration Schema
==================================
The Constitution of Observation (W0 Absolute)

Design Principles:
  - W0 (ObservationEvent) is immutable and precedes all interpretation
  - Segment is the primary observation unit (not token)
  - No routing, no judgment, no meaning in W0
  - Flat schema only (INV-MOL-001 compliance)

Spec: v5.4.0-P9.0
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Literal
from datetime import datetime, timezone
import uuid


# ==========================================
# Constants
# ==========================================

ENGINE_VERSION = "5.4.0-P9.0"


# ==========================================
# Diagnostic Types (Factual Only)
# ==========================================

class DiagnosticType(Enum):
    """
    Diagnostic categories for W0.
    
    CONSTRAINT: These are factual observations about processing,
    NOT interpretations or judgments about meaning.
    """
    NONE = "none"
    TIMEOUT = "timeout"
    PARSE_ERROR = "parse_error"
    ENCODING_ERROR = "encoding_error"
    EMPTY_SEGMENT = "empty_segment"


# ==========================================
# Environment Record
# ==========================================

@dataclass
class EnvironmentRecord:
    """
    Observation environment snapshot.
    
    Purpose: Enable reproducibility by capturing system state
    at the time of observation.
    """
    env_id: str                      # Config hash (determinism)
    
    # --- Observer State ---
    synapse_version: str             # e.g. "v3.0"
    synapse_hash: str                # File hash for audit
    glossary_version: str            # e.g. "v5.3"
    
    # --- Runtime Config ---
    parameters: Dict[str, Any] = field(default_factory=dict)
    # e.g. {"segmenter": "regex_v1", "sentence_boundary": ".!?"}
    
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

@dataclass(frozen=True)
class ObservationEvent:
    """
    W0: Immutable observation record.
    
    IMMUTABILITY: frozen=True ensures that once created, W0 cannot be modified.
    Any attempt to modify fields will raise FrozenInstanceError.
    
    CONSTITUTIONAL ARTICLE 1 (W0 Absolute):
        "The observed fact (W0) takes precedence over any interpretation,
        classification, or meaning assignment. Meaning (W1/W2) is a
        posterior aggregation of W0; W0 itself must not be rewritten."
    
    INVARIANTS:
        - Gateway-issued only (observer = "Gateway")
        - No interpretation, no judgment
        - segment_span references ArticleRecord.raw_text
        - 1 segment = 1 ObservationEvent
    
    WHAT THIS IS NOT:
        - NOT a place for routing decisions (HIT/MISS)
        - NOT a place for meaning or interpretation
        - NOT a place for token-level analysis
    """
    
    # --- Identity ---
    observation_id: str              # UUIDv4 (unique per event)
    source_id: str                   # article_id reference
    timestamp: str                   # ISO8601 (UTC)
    
    # --- Segment Reference ---
    # These reference ArticleRecord.raw_text, NOT a local copy
    segment_index: int               # Processing order (0-indexed)
    segment_span: Tuple[int, int]    # Position in raw_text (start, end)
    
    # Redundant but convenient for debugging (derived, not authoritative)
    segment_text: Optional[str] = None
    
    # --- Observer (Immutable) ---
    observer: Literal["Gateway"] = "Gateway"
    
    # --- Diagnostic (Factual Only) ---
    # CONSTRAINT: diagnostic describes WHAT happened, not WHY or WHAT IT MEANS
    diagnostic_type: DiagnosticType = DiagnosticType.NONE
    diagnostic_detail: Optional[str] = None
    
    # --- Context (Statistics Only) ---
    engine_version: str = ENGINE_VERSION
    context_meta: Dict[str, Any] = field(default_factory=dict)
    # ALLOWED: {"token_count": 5, "char_count": 23}
    # FORBIDDEN: {"meaning": "...", "importance": "high", "routing": "HIT"}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "source_id": self.source_id,
            "timestamp": self.timestamp,
            "segment_index": self.segment_index,
            "segment_span": list(self.segment_span),
            "segment_text": self.segment_text,
            "observer": self.observer,
            "diagnostic_type": self.diagnostic_type.value,
            "diagnostic_detail": self.diagnostic_detail,
            "engine_version": self.engine_version,
            "context_meta": self.context_meta,
        }
    
    @staticmethod
    def generate_id() -> str:
        """Generate unique observation ID."""
        return str(uuid.uuid4())
    
    @staticmethod
    def now_iso() -> str:
        """Generate ISO8601 timestamp (UTC)."""
        return datetime.now(timezone.utc).isoformat()


# ==========================================
# Canonical Molecule (Phase 8 Output + Tracking)
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


@dataclass
class CanonicalMolecule:
    """
    Molecule with tracking metadata.
    
    EXTENSION from Phase 8-3:
        - molecule_id: Deterministic hash for identity
        - source_observation_ids: Links to W0 events
    """
    molecule_id: str                           # Deterministic hash
    source_observation_ids: List[str]          # W0 references (can span multiple)
    formula: str                               # e.g., "aa_1 × aa_2"
    active_atoms: List[CanonicalAtom]
    meta: Dict[str, Any] = field(default_factory=dict)
    # meta["molecule_id_by"] = "integration.gateway" if assigned by integration layer
    
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
# Unknown Token (Phase 7 Output)
# ==========================================

@dataclass
class UnknownToken:
    """
    Token that missed Synapse lookup.
    
    NOTE: This is a placeholder for Phase 7 integration.
    Full schema TBD in Phase 9-1+.
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
    
    NOTE:
        w1_matrix / w2_matrix / aruism_axis are placeholders.
        Final storage location (Article-level vs Cross-Article Index)
        will be decided in Phase 9-1+.
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
    
    # --- Layer 0: Raw ---
    raw_text: str = ""               # Original full text (authoritative)
    segments: List[Tuple[int, int]] = field(default_factory=list)
    # List of (start, end) positions in raw_text
    
    # --- Layer 0.5: Observation Log (W0) ---
    observations: List[ObservationEvent] = field(default_factory=list)
    
    # --- Layer 8: Structure (Strong Path) ---
    molecules: List[CanonicalMolecule] = field(default_factory=list)
    
    # --- Layer 7: Volatility (Weak Path) ---
    unknowns: List[UnknownToken] = field(default_factory=list)
    
    # --- Emergent Slots (Placeholders for Future) ---
    # NO LOGIC IMPLEMENTATION in Phase 9-0
    # These are Optional[Any] = None by design
    w1_matrix: Optional[Any] = None   # W1: Cross-sectional Aggregation
    w2_matrix: Optional[Any] = None   # W2: Conditional Aggregation
    aruism_axis: Optional[Any] = None # Emergent Axis Structure
    
    def __post_init__(self):
        """Initialize defaults and validate."""
        if not self.article_id:
            self.article_id = str(uuid.uuid4())
        if not self.ingestion_time:
            self.ingestion_time = datetime.now(timezone.utc).isoformat()
        
        # Fail-fast: Check molecules for legacy schema
        for mol in self.molecules:
            if isinstance(mol, CanonicalMolecule):
                continue  # Already validated in CanonicalMolecule.__post_init__
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
    
    Ensures consistent ID and timestamp generation.
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
    print("Phase 9-0 Schema Test")
    print("=" * 60)
    
    # Test 1: ObservationEvent creation
    print("\n[Test 1] ObservationEvent creation")
    obs = create_observation_event(
        source_id="article_001",
        segment_index=0,
        segment_span=(0, 15),
        segment_text="I love you.",
        context_meta={"token_count": 3, "char_count": 11},
    )
    print(f"  observation_id: {obs.observation_id}")
    print(f"  observer: {obs.observer}")
    print(f"  segment_span: {obs.segment_span}")
    assert obs.observer == "Gateway"
    print("  ✅ PASS")
    
    # Test 2: Legacy schema detection
    print("\n[Test 2] Legacy schema detection")
    try:
        bad_mol = CanonicalMolecule(
            molecule_id="test",
            source_observation_ids=["obs_1"],
            formula="aa_1",
            active_atoms=[{"coordinates": {"axis": "test"}}],  # BANNED
        )
        print("  ❌ FAIL: Should have raised ValueError")
    except ValueError as e:
        print(f"  Caught: {e}")
        print("  ✅ PASS")
    
    # Test 3: ArticleRecord with observations
    print("\n[Test 3] ArticleRecord creation")
    article = ArticleRecord(
        raw_text="I love you. You love me.",
        segments=[(0, 11), (12, 24)],
    )
    article.observations.append(obs)
    print(f"  article_id: {article.article_id}")
    print(f"  segments: {article.segments}")
    print(f"  observations count: {len(article.observations)}")
    print(f"  get_segment_text(0): '{article.get_segment_text(0)}'")
    assert article.get_segment_text(0) == "I love you."
    print("  ✅ PASS")
    
    # Test 4: Serialization
    print("\n[Test 4] Serialization")
    import json
    article_dict = article.to_dict()
    json_str = json.dumps(article_dict, indent=2, ensure_ascii=False)
    print(f"  JSON length: {len(json_str)} chars")
    assert "observation_id" in json_str
    assert "segment_span" in json_str
    print("  ✅ PASS")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")