"""
ESDE Harvester: Data Collection & Caching Module
=================================================

Separates data collection (I/O) from analysis (Phase 8/9).

Philosophy:
  - "Fetch once, analyze many times"
  - Two-layer storage: Artifact (raw) + Distilled (text)
  - Substrate-compatible traces

Architecture:
  fetcher.py   → HTTP/Wikipedia API access
  distiller.py → Raw response → text + structure stats
  storage.py   → Artifact + dataset file management
  cli.py       → Command-line interface

Usage:
  # Harvest a dataset
  python -m harvester.cli harvest --dataset mixed

  # List cached datasets
  python -m harvester.cli list

  # Run pipeline on cached data (no network)
  python -m statistics.pipeline.run_full_pipeline --dataset mixed

Version: v0.1.0
"""

__version__ = "0.1.0"
