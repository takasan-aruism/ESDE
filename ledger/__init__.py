"""
ESDE Ledger Package
===================
Phase 8-5: Semantic Memory Preconditions
Phase 8-6: Persistent Semantic Ledger (Hash Chain)

Usage:
    # Phase 8-5 (揮発性)
    from ledger import EphemeralLedger
    ledger = EphemeralLedger()
    ledger.upsert(molecule, source_text)
    
    # Phase 8-6 (永続化)
    from ledger import PersistentLedger
    ledger = PersistentLedger("data/semantic_ledger.jsonl")
    ledger.observe_molecule(source_hash, molecule)
"""

# Phase 8-5: Memory Math (既存)
from .memory_math import (
    # Constants
    ALPHA,
    EPSILON,
    DEFAULT_TAU,
    TAU_POLICY,
    VALID_OPERATORS,
    
    # Functions
    decay,
    reinforce,
    should_purge,
    get_tau_from_molecule,
    generate_fingerprint,
    extract_operator_type,
    compute_dt,
    now_utc,
    format_tau,
)

# Phase 8-5: Ephemeral Ledger (既存)
from .ephemeral_ledger import (
    EphemeralLedger,
    MemoryEntry,
    LedgerStats,
)

# Phase 8-6: Canonical JSON (新規)
from .canonical import (
    canonicalize,
    sha256_hex,
    parse_canonical_line,
)

# Phase 8-6: Hash Chain (新規)
from .chain_crypto import (
    GENESIS_PREV,
    compute_hashes,
    compute_event_id,
    compute_self_hash,
)

# Phase 8-6: Persistent Ledger (新規)
from .persistent_ledger import (
    PersistentLedger,
    IntegrityError,
    ValidationReport,
)

__all__ = [
    # Phase 8-5: Classes
    'EphemeralLedger',
    'MemoryEntry',
    'LedgerStats',
    
    # Phase 8-5: Constants
    'ALPHA',
    'EPSILON',
    'DEFAULT_TAU',
    'TAU_POLICY',
    'VALID_OPERATORS',
    
    # Phase 8-5: Functions
    'decay',
    'reinforce',
    'should_purge',
    'get_tau_from_molecule',
    'generate_fingerprint',
    'extract_operator_type',
    'compute_dt',
    'now_utc',
    'format_tau',
    
    # Phase 8-6: Canonical
    'canonicalize',
    'sha256_hex',
    'parse_canonical_line',
    
    # Phase 8-6: Chain
    'GENESIS_PREV',
    'compute_hashes',
    'compute_event_id',
    'compute_self_hash',
    
    # Phase 8-6: Persistent
    'PersistentLedger',
    'IntegrityError',
    'ValidationReport',
]

__version__ = '8.6.0'
