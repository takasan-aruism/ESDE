"""
ESDE Phase 9: Statistics Package (Local)
=========================================

This is the local statistics package for Phase 9 development.
Contains features and pipeline subpackages.

Note: This shadows the Python standard library 'statistics' module
within this package context.
"""

from .features import (
    FeatureExtractor,
    TokenFeature,
    DictionaryProvider,
    NgramCollector,
)

__all__ = [
    "FeatureExtractor",
    "TokenFeature",
    "DictionaryProvider",
    "NgramCollector",
]
