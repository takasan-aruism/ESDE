"""
ESDE Synapse Generator v2.1 (Audit-Compliant Final)

Purpose:
  Connect ESDE semantic atoms to WordNet synsets via embedding similarity.
  This creates the "synapse map" linking ESDE's coordinate system to natural language.

Architecture:
  Phase 1: Collect candidates with LOCAL_TOP_M per synset×concept (explosion prevention)
  Phase 2: Global competition with GLOBAL_TOP_K and diversity constraint (bias prevention)

Key Safety Features (v2.1):
  - LOCAL_TOP_M = 10: Limits candidates per synset×concept pair
  - MAX_PER_CONCEPT = 1: Prevents single concept from dominating a synset's connections
  - MIN_SCORE_THRESHOLD = 0.3: Cuts off weak connections
  - SOFTMAX_TEMP = 0.1: Sharp probability distribution for confident mappings

Input: glossary_results.json (ESDE glossary with axis-level definitions)
Output: esde_synapses_v2_1.json (synset → ESDE atom connections)

Based on: Gemini implementation + GPT audit feedback
"""
import json
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
from datetime import datetime, timezone
import sys
import os


# ==========================================
# Configuration (Audit-Compliant v2.1)
# ==========================================
VERSION = "2.1.0"
INPUT_FILE = "glossary_results.json"
OUTPUT_FILE = "esde_synapses_v2_1.json"
MODEL_NAME = "all-MiniLM-L6-v2"

# Core parameters
GLOBAL_TOP_K = 3          # Max atoms connected to one synset
SOFTMAX_TEMP = 0.1        # Temperature for probability sharpness
MIN_SCORE_THRESHOLD = 0.3 # Minimum similarity to consider

# v2.1 Safety additions
LOCAL_TOP_M = 10          # Max candidates per synset×concept (Phase 1)
MAX_PER_CONCEPT = 1       # Max same-concept entries in Top-K (Phase 2 diversity)


# ==========================================
# Utilities
# ==========================================
def ensure_nltk_data():
    """Download WordNet if not present."""
    try:
        wn.synsets('test')
    except LookupError:
        print("[Setup] Downloading WordNet data...")
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)


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
    
    Args:
        candidates: List sorted by score (descending)
        topk: Maximum number to select
        max_per_concept: Max entries from same concept_id
    
    Returns:
        Diverse top-k list
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
    print(f"=" * 60)
    print(f"ESDE Synapse Generator v{VERSION}")
    print(f"=" * 60)
    print(f"Config:")
    print(f"  GLOBAL_TOP_K: {GLOBAL_TOP_K}")
    print(f"  LOCAL_TOP_M: {LOCAL_TOP_M}")
    print(f"  MAX_PER_CONCEPT: {MAX_PER_CONCEPT}")
    print(f"  SOFTMAX_TEMP: {SOFTMAX_TEMP}")
    print(f"  MIN_SCORE_THRESHOLD: {MIN_SCORE_THRESHOLD}")
    print()
    
    # Load embedding model
    print(f"[1/4] Loading embedding model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    
    # Ensure WordNet
    ensure_nltk_data()
    
    # Load ESDE glossary
    print(f"[2/4] Loading ESDE glossary: {INPUT_FILE} ...")
    if not os.path.exists(INPUT_FILE):
        print(f"[FATAL] File not found: {INPUT_FILE}")
        return
    
    try:
        esde_data = safe_load_esde(INPUT_FILE)
    except Exception as e:
        print(f"[FATAL] Failed to load JSON: {e}")
        return
    
    print(f"  Loaded {len(esde_data)} concepts")
    
    # ---------------------------------------------------------
    # Phase 1: Candidate Collection (with LOCAL_TOP_M)
    # ---------------------------------------------------------
    print(f"\n[3/4] Phase 1: Collecting candidates...")
    
    global_candidates = defaultdict(list)  # synset_id -> candidates
    
    processed = 0
    skipped = 0
    
    for cid, content in esde_data.items():
        # Get final_json or use content directly
        final = content.get("final_json", content)
        
        if not isinstance(final, dict) or "axes" not in final:
            skipped += 1
            continue
        
        name = final.get("name", cid.split(".")[-1] if "." in cid else cid)
        
        # Build search terms
        search_terms = {
            name,
            name.replace("-", "_"),
            name.replace(" ", "_"),
            name.replace("_", " ")
        }
        
        # Collect definitions from all axis-levels
        concept_defs = []
        axes = final.get("axes", {})
        
        if not isinstance(axes, dict):
            skipped += 1
            continue
        
        for axis, levels in axes.items():
            if not isinstance(levels, dict):
                continue
            
            for level, details in levels.items():
                if not isinstance(details, dict):
                    continue
                
                def_text = details.get("definition_en", "")
                if def_text and len(def_text) > 5:
                    concept_defs.append({
                        "ref": f"{cid}:{axis}:{level}",
                        "cid": cid,
                        "axis": axis,
                        "level": level,
                        "text": def_text
                    })
        
        if not concept_defs:
            skipped += 1
            continue
        
        # Find related synsets
        related_synsets = set()
        for term in search_terms:
            for syn in wn.synsets(term):
                related_synsets.add(syn)
        
        if not related_synsets:
            skipped += 1
            continue
        
        # Encode ESDE definitions
        esde_texts = [d["text"] for d in concept_defs]
        esde_embs = model.encode(esde_texts, convert_to_tensor=True)
        
        # Encode WordNet definitions (with examples for context)
        wn_list = list(related_synsets)
        wn_texts = [
            f"{s.definition()} ; {'; '.join(s.examples())}" 
            for s in wn_list
        ]
        wn_embs = model.encode(wn_texts, convert_to_tensor=True)
        
        # Compute similarity matrix
        scores_matrix = util.cos_sim(wn_embs, esde_embs).cpu().numpy()
        
        # Phase 1: LOCAL_TOP_M selection per synset
        for w_idx, syn in enumerate(wn_list):
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
        
        processed += 1
        
        if processed % 50 == 0:
            print(f"  ... processed {processed} concepts")
    
    print(f"  Processed: {processed}, Skipped: {skipped}")
    print(f"  Synsets with candidates: {len(global_candidates)}")
    
    # ---------------------------------------------------------
    # Phase 2: Global Competition & Diversity Selection
    # ---------------------------------------------------------
    print(f"\n[4/4] Phase 2: Global competition & normalization...")
    
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
                "min_threshold": MIN_SCORE_THRESHOLD
            },
            "stats": {
                "input_concepts": len(esde_data),
                "processed_concepts": processed,
                "total_synsets": len(final_registry),
                "total_edges": total_edges,
                "avg_edges_per_synset": round(total_edges / max(len(final_registry), 1), 2)
            }
        },
        "synapses": final_registry
    }
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 60}")
    print(f"COMPLETE")
    print(f"{'=' * 60}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Synsets: {len(final_registry)}")
    print(f"Edges: {total_edges}")
    print(f"Avg edges/synset: {output['_meta']['stats']['avg_edges_per_synset']}")
    print()
    print("Note: Processing time depends on data size and hardware.")
    print("This is an initial synapse map; refinement may be needed based on evaluation.")


if __name__ == "__main__":
    main()
