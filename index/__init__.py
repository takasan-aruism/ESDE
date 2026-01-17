"""
ESDE Index Package
==================
Phase 8-7: Structural Emergence & Semantic Indexing

L2: Semantic Index - L1 (Immutable Ledger) からの射影

Usage:
    from index import QueryAPI
    
    api = QueryAPI(ledger)
    api.get_rigidity("EMO.love")
    api.get_frequency("EMO.love", window=500)
    api.get_recent_directions(limit=100)

Architecture:
    L1 (Ledger) → Projector → L2 (SemanticIndex) → QueryAPI
"""

from .semantic_index import (
    SemanticIndex,
    AtomStats,
    DirectionStats,
    DEFAULT_SEQ_HISTORY_LIMIT,
)

from .projector import (
    Projector,
    create_index_from_ledger,
    extract_formula_signature,
    extract_atoms_from_molecule,
)

from .rigidity import (
    calculate_rigidity,
    calculate_rigidity_windowed,
    get_rigidity_for_atom,
    get_mode_formula,
    classify_rigidity,
    get_all_rigidities,
)

from .query_api import (
    QueryAPI,
    AtomInfo,
    DirectionBalance,
)


__all__ = [
    # Core Classes
    "SemanticIndex",
    "Projector",
    "QueryAPI",
    
    # Data Classes
    "AtomStats",
    "DirectionStats",
    "AtomInfo",
    "DirectionBalance",
    
    # Constants
    "DEFAULT_SEQ_HISTORY_LIMIT",
    
    # Functions - Projector
    "create_index_from_ledger",
    "extract_formula_signature",
    "extract_atoms_from_molecule",
    
    # Functions - Rigidity
    "calculate_rigidity",
    "calculate_rigidity_windowed",
    "get_rigidity_for_atom",
    "get_mode_formula",
    "classify_rigidity",
    "get_all_rigidities",
]

__version__ = "8.7.0"
