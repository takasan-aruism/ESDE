"""
ESDE Synapse Generator v3.0

Purpose:
  Connect ESDE semantic atoms to WordNet synsets via embedding similarity.
  v3.0 extends v2.1 by utilizing triggers_en from Glossary for broader coverage.

Key Changes from v2.1:
  - triggers_en are now included in search terms (phrase + tokenized)
  - anti_triggers_en are explicitly excluded
  - WordNet synset cache for performance
  - Synset embedding cache for performance
  - MAX_SYNSETS_PER_CONCEPT to control explosion
  - --disable_triggers flag for regression testing (v2.1 behavior)

Architecture:
  Phase 1: Collect search terms from name + triggers_en
  Phase 2: WordNet synset discovery with hitcount ranking
  Phase 3: Embedding similarity with LOCAL_TOP_M constraint
  Phase 4: Global competition with GLOBAL_TOP_K and diversity constraint

Input: glossary_results.json (ESDE glossary with axis-level definitions)
Output: esde_synapses_v3.json (synset → ESDE atom connections)

Based on: v2.1 + Gemini design + GPT audit
"""
import json
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
from datetime import datetime, timezone
import sys
import os
import argparse
import re


# ==========================================
# Configuration (v3.0)
# ==========================================
VERSION = "3.0.0"
INPUT_FILE = "glossary_results.json"
OUTPUT_FILE = "esde_synapses_v3.json"
MODEL_NAME = "all-MiniLM-L6-v2"

# Core parameters (maintained from v2.1)
GLOBAL_TOP_K = 3          # Max atoms connected to one synset
SOFTMAX_TEMP = 0.1        # Temperature for probability sharpness
MIN_SCORE_THRESHOLD = 0.3 # Minimum similarity to consider
LOCAL_TOP_M = 10          # Max candidates per synset×concept (Phase 3)
MAX_PER_CONCEPT = 1       # Max same-concept entries in Top-K (Phase 4 diversity)

# v3.0 additions
MAX_SYNSETS_PER_CONCEPT = 500  # Explosion control: max synsets per concept
MIN_WORD_LENGTH = 3            # Minimum word length for tokenized triggers
ALLOWED_POS = {'n', 'v', 'a', 's'}  # Allowed WordNet POS (exclude 'r' adverb)


# ==========================================
# Global Caches
# ==========================================
_synset_cache = {}        # term -> list of synsets
_embedding_cache = {}     # synset_id -> embedding vector
_stopwords = None         # Loaded lazily


# ==========================================
# Utilities
# ==========================================
def ensure_nltk_data():
    """Download required NLTK data if not present."""
    try:
        wn.synsets('test')
    except LookupError:
        print("[Setup] Downloading WordNet data...")
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    
    try:
        stopwords.words('english')
    except LookupError:
        print("[Setup] Downloading stopwords...")
        nltk.download('stopwords', quiet=True)


def get_stopwords():
    """Get English stopwords (cached)."""
    global _stopwords
    if _stopwords is None:
        _stopwords = set(stopwords.words('english'))
        # Add common short words that slip through
        _stopwords.update({'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 
                          'been', 'being', 'have', 'has', 'had', 'do', 'does', 
                          'did', 'will', 'would', 'could', 'should', 'may', 
                          'might', 'must', 'shall', 'can', 'need', 'dare',
                          'ought', 'used', 'to', 'of', 'in', 'for', 'on', 
                          'with', 'at', 'by', 'from', 'as', 'into', 'through',
                          'during', 'before', 'after', 'above', 'below', 'between',
                          'under', 'again', 'further', 'then', 'once', 'here',
                          'there', 'when', 'where', 'why', 'how', 'all', 'each',
                          'few', 'more', 'most', 'other', 'some', 'such', 'no',
                          'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                          'too', 'very', 'just', 'also', 'now', 'new', 'first'})
    return _stopwords


def cached_synsets(term: str) -> list:
    """Get synsets for a term with caching."""
    if term not in _synset_cache:
        try:
            _synset_cache[term] = wn.synsets(term)
        except Exception:
            _synset_cache[term] = []
    return _synset_cache[term]


def lemmatize_word(word: str) -> str:
    """Lemmatize a word using WordNet's morphy."""
    word = word.lower().strip()
    
    # Try morphy for different POS
    for pos in ['n', 'v', 'a']:
        lemma = wn.morphy(word, pos)
        if lemma:
            return lemma
    
    # Fallback: return original
    return word


def is_valid_search_word(word: str) -> bool:
    """Check if a word is valid for search (not stopword, sufficient length, alphabetic)."""
    word = word.lower().strip()
    
    if len(word) < MIN_WORD_LENGTH:
        return False
    
    if not word.isalpha():
        return False
    
    if word in get_stopwords():
        return False
    
    return True


def tokenize_and_filter(phrase: str) -> set:
    """
    Tokenize a phrase and extract valid search words.
    Returns lemmatized content words.
    """
    words = set()
    
    # Split on non-alphabetic characters
    tokens = re.split(r'[^a-zA-Z]+', phrase.lower())
    
    for token in tokens:
        if not token:
            continue
        
        if is_valid_search_word(token):
            # Add original and lemmatized form
            words.add(token)
            lemma = lemmatize_word(token)
            if lemma and lemma != token:
                words.add(lemma)
    
    return words


def extract_search_terms(concept_name: str, axes_data: dict, include_triggers: bool = True) -> set:
    """
    Extract all search terms for a concept.
    
    Args:
        concept_name: The concept name (e.g., "bound")
        axes_data: The axes dictionary from glossary
        include_triggers: If False, only use concept name (v2.1 behavior)
    
    Returns:
        Set of search terms
    """
    search_terms = set()
    
    # Always include concept name and variations
    name_lower = concept_name.lower()
    search_terms.add(name_lower)
    search_terms.add(name_lower.replace("-", "_"))
    search_terms.add(name_lower.replace(" ", "_"))
    search_terms.add(name_lower.replace("_", " "))
    search_terms.add(name_lower.replace("_", ""))
    
    if not include_triggers:
        return search_terms
    
    # Extract triggers_en from all axis-levels
    if not isinstance(axes_data, dict):
        return search_terms
    
    for axis_name, levels in axes_data.items():
        if not isinstance(levels, dict):
            continue
        
        for level_name, details in levels.items():
            if not isinstance(details, dict):
                continue
            
            triggers = details.get("triggers_en", [])
            if not isinstance(triggers, list):
                continue
            
            for trigger in triggers:
                if not isinstance(trigger, str) or not trigger.strip():
                    continue
                
                trigger_lower = trigger.lower().strip()
                
                # Level 1: Add phrase as-is
                search_terms.add(trigger_lower)
                
                # Level 1b: Add underscore version for WordNet multiwords
                search_terms.add(trigger_lower.replace(" ", "_"))
                
                # Level 2 & 3: Tokenize and lemmatize
                words = tokenize_and_filter(trigger)
                search_terms.update(words)
    
    return search_terms


def safe_load_esde(path: str) -> dict:
    """
    Robust loader for ESDE glossary (handles both list and dict formats).
    """
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    
    if isinstance(raw, dict):
        if "glossary" in raw:
            glossary = raw["glossary"]
            if isinstance(glossary, dict):
                return glossary
            elif isinstance(glossary, list):
                return {
                    item["concept_id"]: item 
                    for item in glossary 
                    if isinstance(item, dict) and "concept_id" in item
                }
        return raw
    
    elif isinstance(raw, list):
        return {
            item["concept_id"]: item 
            for item in raw 
            if isinstance(item, dict) and "concept_id" in item
        }
    
    return {}


def softmax(scores: list, temp: float = 1.0) -> list:
    """
    Temperature-scaled softmax.
    Lower temp = sharper distribution (more confident).
    """
    if not scores:
        return []
    
    scores = np.array(scores, dtype=np.float64)
    temp = max(float(temp), 1e-9)
    
    # Numerical stability: subtract max
    exp_scores = np.exp((scores - np.max(scores)) / temp)
    denom = np.sum(exp_scores)
    
    if denom <= 0:
        return [0.0] * len(scores)
    
    return (exp_scores / denom).tolist()


def select_topk_with_diversity(candidates: list, topk: int, max_per_concept: int) -> list:
    """
    Select top-k candidates with diversity constraint.
    Prevents single concept from dominating.
    """
    selected = []
    concept_counts = defaultdict(int)
    
    for c in candidates:
        cid = c.get("concept_id", "")
        
        if concept_counts[cid] >= max_per_concept:
            continue
        
        selected.append(c)
        concept_counts[cid] += 1
        
        if len(selected) >= topk:
            break
    
    return selected


# ==========================================
# Main Processing
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='ESDE Synapse Generator v3.0')
    parser.add_argument('--disable_triggers', action='store_true',
                        help='Disable triggers_en extraction (v2.1 behavior)')
    parser.add_argument('--input', default=INPUT_FILE,
                        help=f'Input glossary file (default: {INPUT_FILE})')
    parser.add_argument('--output', default=OUTPUT_FILE,
                        help=f'Output synapse file (default: {OUTPUT_FILE})')
    parser.add_argument('--max_synsets', type=int, default=MAX_SYNSETS_PER_CONCEPT,
                        help=f'Max synsets per concept (default: {MAX_SYNSETS_PER_CONCEPT})')
    args = parser.parse_args()
    
    include_triggers = not args.disable_triggers
    input_file = args.input
    output_file = args.output
    max_synsets_per_concept = args.max_synsets
    
    print(f"=" * 60)
    print(f"ESDE Synapse Generator v{VERSION}")
    print(f"=" * 60)
    print(f"Config:")
    print(f"  GLOBAL_TOP_K: {GLOBAL_TOP_K}")
    print(f"  LOCAL_TOP_M: {LOCAL_TOP_M}")
    print(f"  MAX_PER_CONCEPT: {MAX_PER_CONCEPT}")
    print(f"  SOFTMAX_TEMP: {SOFTMAX_TEMP}")
    print(f"  MIN_SCORE_THRESHOLD: {MIN_SCORE_THRESHOLD}")
    print(f"  MAX_SYNSETS_PER_CONCEPT: {max_synsets_per_concept}")
    print(f"  Include triggers_en: {include_triggers}")
    print()
    
    # Load embedding model
    print(f"[1/5] Loading embedding model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    
    # Ensure NLTK data
    ensure_nltk_data()
    
    # Load ESDE glossary
    print(f"[2/5] Loading ESDE glossary: {input_file} ...")
    if not os.path.exists(input_file):
        print(f"[FATAL] File not found: {input_file}")
        return
    
    try:
        esde_data = safe_load_esde(input_file)
    except Exception as e:
        print(f"[FATAL] Failed to load JSON: {e}")
        return
    
    print(f"  Loaded {len(esde_data)} concepts")
    
    # Statistics tracking
    stats = {
        "unique_search_terms": 0,
        "total_triggers_extracted": 0,
        "synset_cache_hits": 0,
        "synsets_pruned": 0,
        "concepts_with_no_synsets": 0,
        "avg_terms_per_concept": 0,
        "avg_synsets_before_prune": 0,
        "avg_synsets_after_prune": 0,
    }
    
    all_search_terms = set()
    terms_per_concept = []
    synsets_before_prune = []
    synsets_after_prune = []
    
    # ---------------------------------------------------------
    # Phase 1 & 2: Search term extraction and synset discovery
    # ---------------------------------------------------------
    print(f"\n[3/5] Phase 1-2: Extracting search terms and discovering synsets...")
    
    concept_synset_data = {}  # cid -> {synsets, concept_defs, synset_hitcount}
    
    processed = 0
    skipped = 0
    
    for cid, content in esde_data.items():
        # Get final_json or use content directly
        final = content.get("final_json", content)
        
        if not isinstance(final, dict) or "axes" not in final:
            skipped += 1
            continue
        
        name = final.get("name", cid.split(".")[-1] if "." in cid else cid)
        axes = final.get("axes", {})
        
        if not isinstance(axes, dict):
            skipped += 1
            continue
        
        # Extract search terms
        search_terms = extract_search_terms(name, axes, include_triggers=include_triggers)
        all_search_terms.update(search_terms)
        terms_per_concept.append(len(search_terms))
        
        # Collect definitions from all axis-levels
        concept_defs = []
        for axis_name, levels in axes.items():
            if not isinstance(levels, dict):
                continue
            
            for level_name, details in levels.items():
                if not isinstance(details, dict):
                    continue
                
                def_text = details.get("definition_en", "")
                if def_text and len(def_text) > 5:
                    concept_defs.append({
                        "ref": f"{cid}:{axis_name}:{level_name}",
                        "cid": cid,
                        "axis": axis_name,
                        "level": level_name,
                        "text": def_text
                    })
        
        if not concept_defs:
            skipped += 1
            continue
        
        # Find related synsets with hitcount
        synset_hitcount = defaultdict(int)
        
        for term in search_terms:
            synsets = cached_synsets(term)
            for syn in synsets:
                # POS filter
                if syn.pos() in ALLOWED_POS:
                    synset_hitcount[syn] += 1
        
        if not synset_hitcount:
            stats["concepts_with_no_synsets"] += 1
            skipped += 1
            continue
        
        synsets_before_prune.append(len(synset_hitcount))
        
        # Prune by hitcount: keep top MAX_SYNSETS_PER_CONCEPT
        if len(synset_hitcount) > max_synsets_per_concept:
            sorted_synsets = sorted(synset_hitcount.items(), 
                                   key=lambda x: x[1], reverse=True)
            pruned_synsets = dict(sorted_synsets[:max_synsets_per_concept])
            stats["synsets_pruned"] += len(synset_hitcount) - max_synsets_per_concept
        else:
            pruned_synsets = dict(synset_hitcount)
        
        synsets_after_prune.append(len(pruned_synsets))
        
        concept_synset_data[cid] = {
            "synsets": list(pruned_synsets.keys()),
            "concept_defs": concept_defs,
            "synset_hitcount": pruned_synsets
        }
        
        processed += 1
        
        if processed % 50 == 0:
            print(f"  ... processed {processed} concepts, {len(all_search_terms)} unique terms")
    
    stats["unique_search_terms"] = len(all_search_terms)
    stats["avg_terms_per_concept"] = round(np.mean(terms_per_concept), 2) if terms_per_concept else 0
    stats["avg_synsets_before_prune"] = round(np.mean(synsets_before_prune), 2) if synsets_before_prune else 0
    stats["avg_synsets_after_prune"] = round(np.mean(synsets_after_prune), 2) if synsets_after_prune else 0
    
    print(f"  Processed: {processed}, Skipped: {skipped}")
    print(f"  Unique search terms: {stats['unique_search_terms']}")
    print(f"  Avg terms/concept: {stats['avg_terms_per_concept']}")
    print(f"  Avg synsets before prune: {stats['avg_synsets_before_prune']}")
    print(f"  Avg synsets after prune: {stats['avg_synsets_after_prune']}")
    print(f"  Synsets pruned total: {stats['synsets_pruned']}")
    print(f"  WordNet cache size: {len(_synset_cache)}")
    
    # ---------------------------------------------------------
    # Phase 3: Embedding similarity calculation
    # ---------------------------------------------------------
    print(f"\n[4/5] Phase 3: Computing embedding similarities...")
    
    global_candidates = defaultdict(list)  # synset_id -> candidates
    
    # Pre-compute all unique synset embeddings
    all_synsets = set()
    for data in concept_synset_data.values():
        all_synsets.update(data["synsets"])
    
    print(f"  Total unique synsets to embed: {len(all_synsets)}")
    
    # Batch encode synset definitions
    synset_list = list(all_synsets)
    synset_texts = []
    for syn in synset_list:
        text = f"{syn.definition()} ; {'; '.join(syn.examples())}"
        synset_texts.append(text)
    
    print(f"  Encoding synset definitions...")
    synset_embeddings = model.encode(synset_texts, convert_to_tensor=True, show_progress_bar=True)
    
    # Store in cache
    for i, syn in enumerate(synset_list):
        _embedding_cache[syn.name()] = synset_embeddings[i]
    
    print(f"  Synset embedding cache size: {len(_embedding_cache)}")
    
    # Process each concept
    concept_count = 0
    for cid, data in concept_synset_data.items():
        synsets = data["synsets"]
        concept_defs = data["concept_defs"]
        
        if not synsets or not concept_defs:
            continue
        
        # Encode ESDE definitions
        esde_texts = [d["text"] for d in concept_defs]
        esde_embs = model.encode(esde_texts, convert_to_tensor=True)
        
        # Get synset embeddings from cache
        wn_embs = []
        valid_synsets = []
        for syn in synsets:
            if syn.name() in _embedding_cache:
                wn_embs.append(_embedding_cache[syn.name()])
                valid_synsets.append(syn)
        
        if not wn_embs:
            continue
        
        import torch
        wn_embs_tensor = torch.stack(wn_embs)
        
        # Compute similarity matrix
        scores_matrix = util.cos_sim(wn_embs_tensor, esde_embs).cpu().numpy()
        
        # Phase 3: LOCAL_TOP_M selection per synset
        for w_idx, syn in enumerate(valid_synsets):
            row = scores_matrix[w_idx]
            
            if row.size == 0:
                continue
            
            # Select top-M indices
            m = min(LOCAL_TOP_M, row.size)
            top_idx = np.argpartition(-row, m - 1)[:m]
            top_idx = top_idx[np.argsort(-row[top_idx])]
            
            for e_idx in top_idx:
                score = float(row[e_idx])
                
                if score < MIN_SCORE_THRESHOLD:
                    continue
                
                c_def = concept_defs[int(e_idx)]
                
                candidate = {
                    "concept_id": c_def["cid"],
                    "axis": c_def["axis"],
                    "level": c_def["level"],
                    "score": score,
                    "lemma": syn.lemmas()[0].name(),
                    "pos": syn.pos()
                }
                
                global_candidates[syn.name()].append(candidate)
        
        concept_count += 1
        
        if concept_count % 50 == 0:
            print(f"  ... computed similarities for {concept_count} concepts")
    
    print(f"  Concepts with candidates: {concept_count}")
    print(f"  Synsets with candidates: {len(global_candidates)}")
    
    # ---------------------------------------------------------
    # Phase 4: Global Competition & Diversity Selection
    # ---------------------------------------------------------
    print(f"\n[5/5] Phase 4: Global competition & normalization...")
    
    final_registry = {}  # synset_id -> edges
    total_edges = 0
    
    for syn_id, candidates in global_candidates.items():
        if not candidates:
            continue
        
        # Sort by score (descending)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Apply diversity constraint
        top_k_candidates = select_topk_with_diversity(
            candidates=candidates,
            topk=GLOBAL_TOP_K,
            max_per_concept=MAX_PER_CONCEPT
        )
        
        if not top_k_candidates:
            continue
        
        # Compute softmax weights
        raw_scores = [c["score"] for c in top_k_candidates]
        weights = softmax(raw_scores, temp=SOFTMAX_TEMP)
        
        # Build edges
        edges = []
        for i, cand in enumerate(top_k_candidates):
            edges.append({
                "concept_id": cand["concept_id"],
                "axis": cand["axis"],
                "level": cand["level"],
                "lemma": cand["lemma"],
                "pos": cand["pos"],
                "raw_score": round(cand["score"], 4),
                "weight": round(weights[i], 4),
                "rank": i + 1
            })
        
        final_registry[syn_id] = edges
        total_edges += len(edges)
    
    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------
    output = {
        "_meta": {
            "version": VERSION,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": MODEL_NAME,
            "config": {
                "global_top_k": GLOBAL_TOP_K,
                "local_top_m": LOCAL_TOP_M,
                "max_per_concept": MAX_PER_CONCEPT,
                "softmax_temp": SOFTMAX_TEMP,
                "min_threshold": MIN_SCORE_THRESHOLD,
                "max_synsets_per_concept": max_synsets_per_concept,
                "include_triggers": include_triggers
            },
            "stats": {
                "input_concepts": len(esde_data),
                "processed_concepts": processed,
                "total_synsets": len(final_registry),
                "total_edges": total_edges,
                "avg_edges_per_synset": round(total_edges / max(len(final_registry), 1), 2),
                "unique_search_terms": stats["unique_search_terms"],
                "avg_terms_per_concept": stats["avg_terms_per_concept"],
                "avg_synsets_before_prune": stats["avg_synsets_before_prune"],
                "avg_synsets_after_prune": stats["avg_synsets_after_prune"],
                "synsets_pruned": stats["synsets_pruned"],
                "concepts_with_no_synsets": stats["concepts_with_no_synsets"],
                "synset_cache_size": len(_synset_cache),
                "embedding_cache_size": len(_embedding_cache)
            }
        },
        "synapses": final_registry
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 60}")
    print(f"COMPLETE")
    print(f"{'=' * 60}")
    print(f"Output: {output_file}")
    print(f"Synsets: {len(final_registry)}")
    print(f"Edges: {total_edges}")
    print(f"Avg edges/synset: {output['_meta']['stats']['avg_edges_per_synset']}")
    print()
    print("Stats:")
    for key, value in output['_meta']['stats'].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
