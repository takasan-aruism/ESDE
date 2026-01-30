"""
ESDE Phase 9: Token Feature Extraction Subpackage
==================================================

Provides 20-dimensional token feature vectors and N-gram statistics.

Usage:
    from statistics.features import FeatureExtractor, NgramCollector
    
    # Extract 20-dim features
    extractor = FeatureExtractor()
    features = extractor.extract_text("Oda Nobunaga was born in 1534.")
    
    for feat in features:
        print(f"{feat.token}: {feat.vector}")
    
    # Collect n-grams (separate stream)
    collector = NgramCollector()
    collector.process_tokens_with_features(features)
    bigrams = collector.get_bigram_stats()

Spec: Phase 9 W1 Feature Extraction v1.0
"""

from .dict_provider import (
    DictionaryProvider,
    NULL_SCORE,
    DICT_CONCRETENESS,
    DICT_AOA,
    DICT_SENSORIMOTOR,
    DICT_VALENCE,
    get_concreteness,
    get_aoa,
    get_sensorimotor,
    get_valence,
    DICT_PROVIDER_VERSION,
)

from .feature_extractor import (
    FeatureExtractor,
    TokenFeature,
    FEATURE_NAMES,
    FEATURE_DIM,
    FEATURE_EXTRACTOR_VERSION,
    extract_batch,
)

from .ngram_collector import (
    NgramCollector,
    NgramRecord,
    NGRAM_COLLECTOR_VERSION,
    collect_ngrams_from_text,
)

__version__ = "1.0.0"

__all__ = [
    # Dictionary Provider
    "DictionaryProvider",
    "NULL_SCORE",
    "DICT_CONCRETENESS",
    "DICT_AOA",
    "DICT_SENSORIMOTOR",
    "DICT_VALENCE",
    "get_concreteness",
    "get_aoa",
    "get_sensorimotor",
    "get_valence",
    
    # Feature Extractor
    "FeatureExtractor",
    "TokenFeature",
    "FEATURE_NAMES",
    "FEATURE_DIM",
    "FEATURE_EXTRACTOR_VERSION",
    "extract_batch",
    
    # N-gram Collector
    "NgramCollector",
    "NgramRecord",
    "NGRAM_COLLECTOR_VERSION",
    "collect_ngrams_from_text",
]
