"""
ESDE Phase 9-0: Integration Package
===================================
The Nervous System (神経系の確立)

This package provides the integration layer for ESDE, connecting
the Foundation Layer, Phase 7, and Phase 8 components.

Components:
  - schema.py: Data structures (ArticleRecord, ObservationEvent, etc.)
  - segmenter.py: Text segmentation (sentence, paragraph, hybrid)
  - gateway.py: Content ingestion entry point

Design Philosophy:
  - W0 Absolute: Observation precedes interpretation
  - Segment observation: 1 segment = 1 ObservationEvent
  - No routing logic in Phase 9-0 (boxes and wiring only)

Spec: v5.4.0-P9.0
"""

from .schema import (
    # Enums
    DiagnosticType,
    
    # Core Data Structures
    ObservationEvent,
    EnvironmentRecord,
    CanonicalAtom,
    CanonicalMolecule,
    UnknownToken,
    ArticleRecord,
    
    # Factory Functions
    create_observation_event,
    
    # Constants
    ENGINE_VERSION,
)

from .segmenter import (
    # Base Class
    Segmenter,
    
    # Implementations
    RegexSentenceSegmenter,
    ParagraphSegmenter,
    HybridSegmenter,
    
    # Factory
    get_segmenter,
)

from .gateway import (
    # Config
    GatewayConfig,
    
    # Main Class
    ContentGateway,
)

__all__ = [
    # Schema
    "DiagnosticType",
    "ObservationEvent",
    "EnvironmentRecord",
    "CanonicalAtom",
    "CanonicalMolecule",
    "UnknownToken",
    "ArticleRecord",
    "create_observation_event",
    "ENGINE_VERSION",
    
    # Segmenter
    "Segmenter",
    "RegexSentenceSegmenter",
    "ParagraphSegmenter",
    "HybridSegmenter",
    "get_segmenter",
    
    # Gateway
    "GatewayConfig",
    "ContentGateway",
]

__version__ = "5.4.0-P9.0"
