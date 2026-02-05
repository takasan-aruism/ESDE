"""
ESDE Integration - Relations Package
=====================================
Observation C: Deterministic relation extraction via spaCy SVO.

Phase 8 ↔ Phase 9 bridge layer.
Produces relation edges (JSONL) from text without LLM.

Modules:
  parser_adapter.py   - spaCy dependency → SVO triples
  relation_logger.py  - Synapse grounding → relation edges
"""

__version__ = "0.1.0"