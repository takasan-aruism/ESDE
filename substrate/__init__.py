"""
ESDE Substrate Layer (Context Fabric)
=====================================

Layer 0: The observation substrate beneath all ESDE phases.

Philosophy: "Describe, but do not decide."

This layer provides:
  - ContextRecord: Immutable observation records
  - SubstrateRegistry: Append-only storage
  - Trace normalization and validation

Design Principles:
  - No semantic interpretation
  - Machine-observable facts only
  - Deterministic ID generation
  - Append-only storage

Invariants:
  INV-SUB-001: Upper Read-Only
  INV-SUB-002: No Semantic Transform
  INV-SUB-003: Machine-Observable
  INV-SUB-004: No Inference
  INV-SUB-005: Append-Only
  INV-SUB-006: ID Determinism
  INV-SUB-007: Canonical Export

Usage:
    from substrate import (
        SubstrateRegistry,
        create_context_record,
        ContextRecord,
    )
    
    # Create a registry
    registry = SubstrateRegistry()
    
    # Register traces
    context_id = registry.register_traces(
        traces={"html:tag_count": 42, "text:char_count": 1000},
        retrieval_path="https://example.com/article",
    )
    
    # Retrieve record
    record = registry.get(context_id)

Spec: Substrate Layer v0.1.0
"""

# Version
__version__ = "0.1.0"

# Schema
from .schema import (
    ContextRecord,
    create_context_record,
    convert_source_meta_to_traces,
    SUBSTRATE_SCHEMA_VERSION,
    DEFAULT_CAPTURE_VERSION,
)

# Registry
from .registry import (
    SubstrateRegistry,
    REGISTRY_VERSION,
    DEFAULT_REGISTRY_PATH,
)

# ID Generator
from .id_generator import (
    compute_context_id,
    verify_context_id,
    canonical_json_dumps,
    ID_GENERATOR_VERSION,
    CONTEXT_ID_LENGTH,
)

# Traces
from .traces import (
    validate_trace_key,
    validate_trace_value,
    validate_traces,
    normalize_traces,
    normalize_float,
    normalize_value,
    extract_namespace,
    group_by_namespace,
    TRACE_NORMALIZER_VERSION,
    FLOAT_PRECISION,
    FORBIDDEN_NAMESPACE_PREFIXES,
    FORBIDDEN_KEY_NAMES,
    ALLOWED_KEY_NAME_EXCEPTIONS,
    ALLOWED_VALUE_TYPES,
    STRING_MAX_LENGTH,
    INT_MIN,
    INT_MAX,
)

# Public API
__all__ = [
    # Version
    "__version__",
    
    # Schema (Primary API)
    "ContextRecord",
    "create_context_record",
    "convert_source_meta_to_traces",
    "SUBSTRATE_SCHEMA_VERSION",
    "DEFAULT_CAPTURE_VERSION",
    
    # Registry (Primary API)
    "SubstrateRegistry",
    "REGISTRY_VERSION",
    "DEFAULT_REGISTRY_PATH",
    
    # ID Generator
    "compute_context_id",
    "verify_context_id",
    "canonical_json_dumps",
    "ID_GENERATOR_VERSION",
    "CONTEXT_ID_LENGTH",
    
    # Traces (Validation & Normalization)
    "validate_trace_key",
    "validate_trace_value",
    "validate_traces",
    "normalize_traces",
    "normalize_float",
    "normalize_value",
    "extract_namespace",
    "group_by_namespace",
    "TRACE_NORMALIZER_VERSION",
    "FLOAT_PRECISION",
    "FORBIDDEN_NAMESPACE_PREFIXES",
    "FORBIDDEN_KEY_NAMES",
    "ALLOWED_KEY_NAME_EXCEPTIONS",
    "ALLOWED_VALUE_TYPES",
    "STRING_MAX_LENGTH",
    "INT_MIN",
    "INT_MAX",
]
