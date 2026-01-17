"""
ESDE Engine v5.3.2 - Utility Functions
"""
import re
import os
import math
import hashlib
import secrets
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timezone

import nltk
from nltk.corpus import wordnet as wn

from .config import (
    STOPWORDS, MIN_TOKEN_LENGTH, TYPO_MAX_EDIT_DISTANCE, 
    TYPO_MIN_CONFIDENCE, TYPO_REFERENCE_WORDS, PROPER_NOUN_INDICATORS
)


def ensure_nltk_data():
    """Ensure WordNet data is available."""
    try:
        wn.synsets('test')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    tokens = text.split()
    return [t.strip("'") for t in tokens if t.strip("'")]


def is_stopword_or_noise(token: str) -> bool:
    """Check if token should be COMPLETELY SKIPPED."""
    if len(token) < MIN_TOKEN_LENGTH:
        return True
    return token in STOPWORDS


def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of a file for audit trail."""
    if not os.path.exists(filepath):
        return "file_not_found"
    try:
        with open(filepath, 'rb') as f:
            return f"sha256:{hashlib.sha256(f.read()).hexdigest()[:16]}"
    except Exception:
        return "hash_error"


def generate_run_id() -> str:
    """Generate run_id: YYYYMMDD_HHMMSS_<random6>"""
    now = datetime.now(timezone.utc)
    random_suffix = secrets.token_hex(3)
    return f"{now.strftime('%Y%m%d_%H%M%S')}_{random_suffix}"


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def find_typo_candidate(token: str) -> Optional[Dict[str, Any]]:
    """
    Route A: Check if token is likely a typo of a known word.
    Only returns candidates with distance > 0 (actual typos, not exact matches).
    """
    if len(token) < 3:
        return None
    
    best_match = None
    best_distance = float('inf')
    
    for ref_word in TYPO_REFERENCE_WORDS:
        dist = levenshtein_distance(token, ref_word)
        if 0 < dist <= TYPO_MAX_EDIT_DISTANCE and dist < best_distance:
            best_distance = dist
            best_match = ref_word
    
    if best_match:
        len_diff = abs(len(token) - len(best_match))
        max_len = max(len(token), len(best_match))
        
        if best_distance == 1 and len_diff <= 1:
            confidence = 0.85 - (len_diff * 0.1)
        else:
            len_factor = 1.0 - (len_diff / max_len)
            confidence = (1.0 - best_distance / 3.0) * len_factor
        
        if confidence >= TYPO_MIN_CONFIDENCE:
            return {
                "original": token,
                "suggestion": best_match,
                "distance": best_distance,
                "confidence": round(confidence, 3)
            }
    return None


def find_all_typo_candidates(token: str, max_candidates: int = 3) -> List[Dict[str, Any]]:
    """
    Find ALL typo candidates for queue recording (even below threshold).
    Used for Route B tokens to record potential typo corrections.
    """
    if len(token) < 3 or len(token) > 6:
        return []
    
    candidates = []
    
    for ref_word in TYPO_REFERENCE_WORDS:
        dist = levenshtein_distance(token, ref_word)
        if 0 < dist <= TYPO_MAX_EDIT_DISTANCE:
            if wn.synsets(ref_word):
                len_diff = abs(len(token) - len(ref_word))
                max_len = max(len(token), len(ref_word))
                
                if dist == 1 and len_diff <= 1:
                    confidence = 0.85 - (len_diff * 0.1)
                else:
                    len_factor = 1.0 - (len_diff / max_len)
                    confidence = (1.0 - dist / 3.0) * len_factor
                
                candidates.append({
                    "candidate": ref_word,
                    "dist": dist,
                    "confidence": round(confidence, 3)
                })
    
    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    return candidates[:max_candidates]


def compute_dedup_key(token_norm: str, pos_guess: str, top_synset: Optional[str]) -> str:
    """Compute dedup key for unknown queue records."""
    pos = pos_guess if pos_guess else "u"
    synset = top_synset if top_synset else "none"
    key_str = f"{token_norm}|{pos}|{synset}"
    return hashlib.sha1(key_str.encode('utf-8')).hexdigest()[:16]


def is_proper_noun_candidate(synset_ids: List[str]) -> bool:
    """Route B: Check if token's synsets suggest it's a proper noun."""
    for sid in synset_ids:
        try:
            synset = wn.synset(sid)
            definition = synset.definition().lower()
            for indicator in PROPER_NOUN_INDICATORS:
                if indicator in definition:
                    return True
            for hyper in synset.hypernyms()[:5]:
                hyper_name = hyper.name().split('.')[0]
                if hyper_name in PROPER_NOUN_INDICATORS:
                    return True
        except Exception:
            continue
    return False


def guess_category(token: str, synset_ids: List[str], has_edges: bool,
                   typo_candidates: List[Dict]) -> tuple:
    """Guess category for queue record (provisional, not final)."""
    if typo_candidates and typo_candidates[0].get("confidence", 0) >= 0.5:
        return ("typo_like", "spellcheck", typo_candidates[0]["confidence"])
    
    if synset_ids and is_proper_noun_candidate(synset_ids):
        return ("proper_noun_like", "web", 0.7)
    
    if not synset_ids:
        if len(token) <= 4:
            return ("slang_like", "web", 0.5)
        else:
            return ("unknown", "molecule", 0.3)
    
    if synset_ids and not has_edges:
        return ("unknown", "molecule", 0.4)
    
    return ("unknown", "ignore", None)


def compute_entropy(weights: List[float]) -> float:
    """Compute entropy of weight distribution."""
    if not weights or len(weights) < 2:
        return 0.0
    total = sum(weights)
    if total <= 0:
        return 0.0
    normalized = [w / total for w in weights]
    entropy = 0.0
    for p in normalized:
        if p > 0:
            entropy -= p * math.log(p)
    return entropy


def compute_variance_metrics(weights: List[float], 
                              margin_threshold: float,
                              entropy_threshold: float) -> Dict[str, Any]:
    """Compute variance metrics for edge weight distribution."""
    if not weights:
        return {"margin": 1.0, "entropy": 0.0, "high_variance": False}
    sorted_w = sorted(weights, reverse=True)
    top1 = sorted_w[0] if len(sorted_w) > 0 else 0.0
    top2 = sorted_w[1] if len(sorted_w) > 1 else 0.0
    margin = top1 - top2
    entropy = compute_entropy(weights)
    high_variance = (margin < margin_threshold or entropy > entropy_threshold)
    return {"margin": round(margin, 4), "entropy": round(entropy, 4), "high_variance": high_variance}
