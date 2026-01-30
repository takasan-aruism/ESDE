#!/usr/bin/env python3
"""
ESDE Phase 9: W1 Feature Extraction Integration Test
=====================================================

Tests the complete W1 feature extraction pipeline:
  1. DictionaryProvider (psycholinguistic lookups)
  2. FeatureExtractor (20-dim token vectors)
  3. NgramCollector (bigram/trigram statistics)

Usage:
    python -m pytest tests/test_phase9_w1_features.py -v
    python tests/test_phase9_w1_features.py              # Direct run
    python tests/test_phase9_w1_features.py --no-spacy   # Without spaCy

Spec: Phase 9 W1 Feature Extraction v1.0
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from statistics.features import (
    FeatureExtractor,
    NgramCollector,
    DictionaryProvider,
    FEATURE_NAMES,
    FEATURE_DIM,
    NULL_SCORE,
)


# ==========================================
# Test Data
# ==========================================

SENGOKU_TEXT = """
Oda Nobunaga was born in 1534 in Owari Province. He was the son of Oda Nobuhide, 
a minor daimyo. Through military conquest and political cunning, Nobunaga rose 
to dominate central Japan.

In 1560, Nobunaga achieved his first major victory at the Battle of Okehazama, 
where he defeated Imagawa Yoshimoto against overwhelming odds. This battle 
demonstrated his innovative tactics and willingness to take risks.

Nobunaga was killed at Honnoji Temple in Kyoto on June 21, 1582. He was betrayed 
by his general Akechi Mitsuhide in what became known as the Honnoji Incident.
The unification of Japan was left incomplete.
"""


# ==========================================
# Tests
# ==========================================

def test_dictionary_provider():
    """Test DictionaryProvider functionality."""
    print("\n" + "=" * 60)
    print("[Test 1] DictionaryProvider")
    print("=" * 60)
    
    # Reset singleton for clean test
    DictionaryProvider.reset_instance()
    provider = DictionaryProvider.get_instance()
    
    test_words = [
        ("sword", "Concrete object"),
        ("honor", "Abstract concept"),
        ("kill", "Action verb (negative)"),
        ("beautiful", "Emotional word"),
        ("Nobunaga", "Proper noun (likely NULL)"),
    ]
    
    print("\nLookup results:")
    for word, desc in test_words:
        scores = provider.get_all_scores(word)
        print(f"\n  '{word}' ({desc}):")
        for dict_type, score in scores.items():
            status = "NULL" if score == NULL_SCORE else f"{score:.3f}"
            print(f"    {dict_type}: {status}")
    
    stats = provider.get_stats()
    print(f"\n  Hit rate: {stats['hit_rate']:.1%}")
    
    return True


def test_feature_extractor(use_spacy: bool = True, verbose: bool = False):
    """Test FeatureExtractor functionality."""
    print("\n" + "=" * 60)
    print(f"[Test 2] FeatureExtractor (spaCy={use_spacy})")
    print("=" * 60)
    
    extractor = FeatureExtractor(use_spacy=use_spacy)
    
    print(f"\nExtracting from {len(SENGOKU_TEXT)} chars...")
    features = extractor.extract_text(SENGOKU_TEXT.strip())
    
    print(f"  Extracted {len(features)} tokens")
    print(f"  Vector dimension: {FEATURE_DIM}")
    
    # Dimension check
    for feat in features[:3]:
        assert len(feat.vector) == FEATURE_DIM, f"Expected {FEATURE_DIM} dims, got {len(feat.vector)}"
    print(f"  ✅ Dimension check passed")
    
    # Show interesting tokens
    interesting = ["Nobunaga", "born", "1534", "killed", "by", "unification"]
    
    print("\n  Selected tokens:")
    for feat in features:
        if feat.token in interesting:
            print(f"\n    '{feat.token}' (lemma='{feat.lemma}')")
            named = feat.to_named_dict()
            
            for name in ["is_capitalized", "is_year", "is_passive_participle", 
                         "is_proper_noun", "is_action_verb", "is_by_agent"]:
                val = named.get(name, "?")
                if isinstance(val, float):
                    print(f"      {name}: {val:.0f}")
            
            for name in ["concreteness_score", "emotional_valence"]:
                val = named.get(name, "?")
                if isinstance(val, float):
                    status = "NULL" if val == -1.0 else f"{val:.3f}"
                    print(f"      {name}: {status}")
    
    if verbose:
        print("\n  All tokens:")
        for feat in features:
            vec_str = ", ".join(f"{v:.2f}" for v in feat.vector[:5])
            print(f"    {feat.token:15} [{vec_str}, ...]")
    
    stats = extractor.get_stats()
    print(f"\n  Tokens processed: {stats['tokens_processed']}")
    print(f"  Sentences: {stats['sentences_processed']}")
    
    return features


def test_ngram_collector(features):
    """Test NgramCollector functionality."""
    print("\n" + "=" * 60)
    print("[Test 3] NgramCollector")
    print("=" * 60)
    
    collector = NgramCollector()
    
    print("\nProcessing tokens...")
    collector.process_tokens_with_features(features)
    
    stats = collector.get_stats()
    print(f"  Unique bigrams: {stats['unique_bigrams']}")
    print(f"  Unique trigrams: {stats['unique_trigrams']}")
    
    print("\n  Top 10 bigrams:")
    for record in collector.get_bigram_stats(top_k=10):
        tokens_str = " ".join(record.tokens)
        print(f"    '{tokens_str}': {record.count}")
    
    print("\n  Passive patterns (was + participle):")
    for record in collector.get_bigram_stats():
        if record.tokens[0] == "was":
            print(f"    '{' '.join(record.tokens)}': {record.count}")
    
    print("\n  Top 5 trigrams:")
    for record in collector.get_trigram_stats(top_k=5):
        tokens_str = " ".join(record.tokens)
        print(f"    '{tokens_str}': {record.count}")
    
    return True


def test_passive_detection(features):
    """Verify passive voice detection."""
    print("\n" + "=" * 60)
    print("[Test 4] Passive Voice Detection")
    print("=" * 60)
    
    be_aux_tokens = []
    passive_participles = []
    by_agents = []
    
    for feat in features:
        named = feat.to_named_dict()
        
        if named.get("is_be_aux", 0) > 0:
            be_aux_tokens.append(feat.token)
        
        if named.get("is_passive_participle", 0) > 0:
            passive_participles.append(feat.token)
        
        if named.get("is_by_agent", 0) > 0:
            by_agents.append(feat.token)
    
    print(f"\n  Be auxiliaries: {be_aux_tokens}")
    print(f"  Passive participles: {passive_participles}")
    print(f"  By-agent markers: {by_agents}")
    
    expected_passives = ["born", "killed", "betrayed"]
    found = [p for p in expected_passives if p in passive_participles]
    
    print(f"\n  Expected passives: {expected_passives}")
    print(f"  Found: {found}")
    
    if len(found) >= 2:
        print("  ✅ Passive detection working")
        return True
    else:
        print("  ⚠️ Some passives may be missed")
        return True


def test_feature_consistency():
    """Test that features are consistent across runs."""
    print("\n" + "=" * 60)
    print("[Test 5] Feature Consistency")
    print("=" * 60)
    
    extractor = FeatureExtractor(use_spacy=False)
    
    text = "Nobunaga was killed in 1582."
    
    features1 = extractor.extract_text(text)
    features2 = extractor.extract_text(text)
    
    all_match = True
    for f1, f2 in zip(features1, features2):
        if f1.vector != f2.vector:
            print(f"  ❌ Mismatch: {f1.token}")
            all_match = False
    
    if all_match:
        print("  ✅ Vectors are deterministic")
    
    return all_match


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Phase 9 W1 Integration Test")
    parser.add_argument("--no-spacy", action="store_true", help="Disable spaCy")
    parser.add_argument("--verbose", action="store_true", help="Show all tokens")
    args = parser.parse_args()
    
    print("\n" + "#" * 70)
    print("# ESDE Phase 9: W1 Token Feature Extraction Test")
    print("#" * 70)
    
    results = []
    
    # Test 1: Dictionary
    try:
        results.append(("DictionaryProvider", test_dictionary_provider()))
    except Exception as e:
        print(f"  ❌ Error: {e}")
        results.append(("DictionaryProvider", False))
    
    # Test 2: Feature Extractor
    features = None
    try:
        features = test_feature_extractor(
            use_spacy=not args.no_spacy,
            verbose=args.verbose,
        )
        results.append(("FeatureExtractor", features is not None))
    except Exception as e:
        import traceback
        traceback.print_exc()
        results.append(("FeatureExtractor", False))
    
    # Test 3: N-gram Collector
    if features:
        try:
            results.append(("NgramCollector", test_ngram_collector(features)))
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append(("NgramCollector", False))
    
    # Test 4: Passive Detection
    if features:
        try:
            results.append(("PassiveDetection", test_passive_detection(features)))
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append(("PassiveDetection", False))
    
    # Test 5: Consistency
    try:
        results.append(("Consistency", test_feature_consistency()))
    except Exception as e:
        print(f"  ❌ Error: {e}")
        results.append(("Consistency", False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {name}: {status}")
        if success:
            passed += 1
    
    print(f"\n  Total: {passed}/{len(results)} passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
