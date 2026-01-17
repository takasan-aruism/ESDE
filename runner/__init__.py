"""
ESDE Runner Package
===================
Phase 8-9: Semantic Monitor & Continuous Existence

長時間稼働検証ハーネス。
"""

from .long_run import (
    LongRunRunner,
    LongRunReport,
    DEFAULT_CORPUS,
    print_report,
)


__all__ = [
    "LongRunRunner",
    "LongRunReport",
    "DEFAULT_CORPUS",
    "print_report",
]

__version__ = "8.9.0"
