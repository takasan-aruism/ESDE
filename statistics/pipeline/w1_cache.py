"""
ESDE Phase 9: W1 Feature Cache
================================

Caches W1 feature extraction results to avoid repeated spaCy processing.

W1 output (TokenFeature list + sections) depends ONLY on:
  1. Source text content (hash)
  2. FeatureExtractor version
  3. spaCy model name

Changing lens/axis/threshold does NOT affect W1 → safe to cache.

Storage: data/features/{dataset}/
  manifest.json  — version info + per-article text hashes
  {article_id}.json — serialized features + sections

"Describe, but do not decide" — cache stores raw observations.
"""

import json
import hashlib
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict

from statistics.features.feature_extractor import (
    TokenFeature,
    FEATURE_EXTRACTOR_VERSION,
    FEATURE_DIM,
)


CACHE_VERSION = "1.0"
FEATURES_DIR = Path("data/features")


def _text_hash(text: str) -> str:
    """Compute SHA-256 hash of text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _serialize_feature(f: TokenFeature) -> Dict[str, Any]:
    """Serialize a TokenFeature to JSON-compatible dict."""
    return {
        "token": f.token,
        "lemma": f.lemma,
        "vector": list(f.vector),
        "token_idx": f.token_idx,
        "sentence_idx": f.sentence_idx,
        "char_start": f.char_start,
        "char_end": f.char_end,
        "section_idx": f.section_idx,
        "section_name": f.section_name,
        "paragraph_idx": f.paragraph_idx,
        "paragraph_count": f.paragraph_count,
        "sentences_in_paragraph": f.sentences_in_paragraph,
        "sentence_length": f.sentence_length,
        "total_sentences": f.total_sentences,
        "total_sections": f.total_sections,
    }


def _deserialize_feature(d: Dict[str, Any]) -> TokenFeature:
    """Deserialize a dict back to TokenFeature."""
    return TokenFeature(
        token=d["token"],
        lemma=d["lemma"],
        vector=tuple(d["vector"]),
        token_idx=d.get("token_idx", 0),
        sentence_idx=d.get("sentence_idx", 0),
        char_start=d.get("char_start", 0),
        char_end=d.get("char_end", 0),
        section_idx=d.get("section_idx", 0),
        section_name=d.get("section_name", ""),
        paragraph_idx=d.get("paragraph_idx", 0),
        paragraph_count=d.get("paragraph_count", 1),
        sentences_in_paragraph=d.get("sentences_in_paragraph", 1),
        sentence_length=d.get("sentence_length", 0),
        total_sentences=d.get("total_sentences", 1),
        total_sections=d.get("total_sections", 1),
    )


def _cache_dir(dataset: str) -> Path:
    """Get cache directory for a dataset."""
    return FEATURES_DIR / dataset


def _load_manifest(dataset: str) -> Optional[Dict[str, Any]]:
    """Load cache manifest if it exists."""
    path = _cache_dir(dataset) / "manifest.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_manifest(dataset: str, manifest: Dict[str, Any]):
    """Save cache manifest."""
    d = _cache_dir(dataset)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def check_cache(
    dataset: str,
    articles: Dict[str, str],
    spacy_model: str = "en_core_web_sm",
) -> Tuple[bool, str]:
    """
    Check if cached features are valid for the given articles.
    
    Returns:
        (is_valid, reason)
    """
    manifest = _load_manifest(dataset)
    
    if manifest is None:
        return False, "no cache found"
    
    # Version check
    if manifest.get("cache_version") != CACHE_VERSION:
        return False, f"cache version mismatch ({manifest.get('cache_version')} != {CACHE_VERSION})"
    
    if manifest.get("extractor_version") != FEATURE_EXTRACTOR_VERSION:
        return False, f"extractor version mismatch ({manifest.get('extractor_version')} != {FEATURE_EXTRACTOR_VERSION})"
    
    if manifest.get("spacy_model") != spacy_model:
        return False, f"spaCy model mismatch ({manifest.get('spacy_model')} != {spacy_model})"
    
    # Article-level hash check
    cached_articles = manifest.get("articles", {})
    
    for article_id, text in articles.items():
        cached = cached_articles.get(article_id)
        if cached is None:
            return False, f"article '{article_id}' not in cache"
        
        current_hash = _text_hash(text)
        if cached.get("text_hash") != current_hash:
            return False, f"article '{article_id}' text changed"
        
        # Check that the feature file exists
        feature_path = _cache_dir(dataset) / f"{article_id}.json"
        if not feature_path.exists():
            return False, f"feature file missing for '{article_id}'"
    
    return True, f"cache valid ({len(articles)} articles)"


def save_cache(
    dataset: str,
    articles: Dict[str, str],
    all_features: Dict[str, List[TokenFeature]],
    all_sections: Dict[str, List[Dict[str, Any]]],
    spacy_model: str = "en_core_web_sm",
):
    """
    Save W1 extraction results to cache.
    
    Args:
        dataset: Dataset name
        articles: {article_id: raw_text}
        all_features: {article_id: [TokenFeature, ...]}
        all_sections: {article_id: [{title, level, content}, ...]}
        spacy_model: spaCy model used
    """
    cache_dir = _cache_dir(dataset)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = {
        "cache_version": CACHE_VERSION,
        "extractor_version": FEATURE_EXTRACTOR_VERSION,
        "feature_dim": FEATURE_DIM,
        "spacy_model": spacy_model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "article_count": len(articles),
        "articles": {},
    }
    
    for article_id, text in articles.items():
        features = all_features.get(article_id, [])
        sections = all_sections.get(article_id, [])
        
        # Save per-article feature file
        article_data = {
            "article_id": article_id,
            "token_count": len(features),
            "section_count": len(sections),
            "sections": sections,
            "features": [_serialize_feature(f) for f in features],
        }
        
        feature_path = cache_dir / f"{article_id}.json"
        with open(feature_path, "w", encoding="utf-8") as f:
            json.dump(article_data, f, ensure_ascii=False)
        
        # Manifest entry
        manifest["articles"][article_id] = {
            "text_hash": _text_hash(text),
            "token_count": len(features),
            "section_count": len(sections),
        }
    
    _save_manifest(dataset, manifest)
    
    total_tokens = sum(len(f) for f in all_features.values())
    print(f"  [Cache] Saved {len(articles)} articles, {total_tokens:,} tokens → {cache_dir}")


def load_cache(
    dataset: str,
    article_ids: List[str],
) -> Tuple[Dict[str, List[TokenFeature]], Dict[str, List[Dict[str, Any]]]]:
    """
    Load W1 extraction results from cache.
    
    Args:
        dataset: Dataset name
        article_ids: Articles to load
        
    Returns:
        (all_features, all_sections)
    """
    cache_dir = _cache_dir(dataset)
    
    all_features: Dict[str, List[TokenFeature]] = {}
    all_sections: Dict[str, List[Dict[str, Any]]] = {}
    
    for article_id in article_ids:
        feature_path = cache_dir / f"{article_id}.json"
        
        with open(feature_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        features = [_deserialize_feature(d) for d in data["features"]]
        sections = data["sections"]
        
        all_features[article_id] = features
        all_sections[article_id] = sections
    
    total_tokens = sum(len(f) for f in all_features.values())
    print(f"  [Cache] Loaded {len(article_ids)} articles, {total_tokens:,} tokens ← {cache_dir}")
    
    return all_features, all_sections
