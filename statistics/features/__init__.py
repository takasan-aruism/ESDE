"""
ESDE Phase 9: Token Feature Extraction Subpackage
==================================================

Provides 20-dimensional token feature vectors, N-gram statistics,
and structure statistics.

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
    
    # Structure statistics
    from statistics.features import compute_structure_stats
    stats = compute_structure_stats(features)
    print(f"Avg sentence length: {stats.avg_sentence_length}")

Spec: Phase 9 W1 Feature Extraction v1.1
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

from .structure_stats import (
    StructureStats,
    compute_structure_stats,
    compute_article_stats,
    compare_articles,
)

__version__ = "1.1.0"

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
    
    # Structure Statistics
    "StructureStats",
    "compute_structure_stats",
    "compute_article_stats",
    "compare_articles",
]
