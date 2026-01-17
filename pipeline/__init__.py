"""
ESDE Pipeline Package
=====================
Phase 8-8: Live Integration & Introspective Feedback

End-to-End統合パイプライン。
"""

from .core_pipeline import (
    ESDEPipeline,
    PipelineResult,
    ModulatedGenerator,
)


__all__ = [
    "ESDEPipeline",
    "PipelineResult",
    "ModulatedGenerator",
]

__version__ = "8.8.0"
