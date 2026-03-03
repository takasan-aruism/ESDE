#!/usr/bin/env python3
"""
ESDE — Synapse v4 Comparison Experiment
========================================
Validates existing Synapse (embedding-based) against A1 empirical 48D measurements.
Analysis-only. No Synapse modification.

Per GPT instruction document (2026-03-01):
  Task 1: Atom Centroid Construction
  Task 2: Word-to-Atom Distance Evaluation
  Task 3: Synapse Edge Comparison
  Task 4: Statistical Output

Usage:
  python3 synapse_v4_compare.py \
    --a1-dir audit_output/ \
    --lexicon-dir lexicon/ \
    --synapse esde_synapses_v3.json \
    --patches patches/synapse_v3.1.json patches/synapse_v3.2.json \
    --dictionary esde_dictionary.json \
    --out-dir synapse_v4_report/
"""

import json
import math
import argparse
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timezone


# ============================================================
# 48 Slot IDs (canonical order from mapper_a1.py)
# ============================================================

SLOT_IDS = [
    "temporal.emergence", "temporal.establishment", "temporal.peak",
    "temporal.decline", "temporal.dissolution",
    "scale.individual", "scale.group", "scale.institution",
    "scale.civilization", "scale.universal",
    "ontological.material", "ontological.informational",
    "ontological.relational", "ontological.structural", "ontological.semantic",
    "interconnection.independent", "interconnection.catalytic",
    "interconnection.chained", "interconnection.synchronous", "interconnection.resonant",
    "resonance.superficial", "resonance.structural",
    "resonance.essential", "resonance.existential",
    "symmetry.destructive", "symmetry.inclusive",
    "symmetry.transformative", "symmetry.generative", "symmetry.cyclical",
    "lawfulness.causal", "lawfulness.emergent",
    "lawfulness.necessary", "lawfulness.contingent",
    "agency.reactive", "agency.adaptive",
    "agency.intentional", "agency.autonomous",
    "boundary.permeable", "boundary.selective",
    "boundary.rigid", "boundary.dissolved",
    "potential.latent", "potential.activated",
    "potential.kinetic", "potential.exhausted",
    "identity.generic", "identity.specific",
    "identity.archetypal",
]

assert len(SLOT_IDS) == 48, f"Expected 48 slots, got {len(SLOT_IDS)}"

# ============================================================
# Thresholds (GPT instruction — temporary for analysis only)
# ============================================================

EMBEDDING_HIGH_TH = 0.55
A1_COSINE_HIGH_TH = 0.65
MARGIN_HIGH_TH = 0.05


# ============================================================
# Math Utilities
# ============================================================

def vec_from_scores(scores: Dict[str, float]) -> List[float]:
    """Extract 48D vector from score dict, in canonical slot order."""
    return [scores.get(s, 0.0) for s in SLOT_IDS]


def cosine_sim(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def l2_dist(a: List[float], b: List[float]) -> float:
    """Euclidean distance between two vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def vec_mean(vectors: List[List[float]]) -> List[float]:
    """Element-wise mean of a list of vectors."""
    if not vectors:
        return [0.0] * 48
    n = len(vectors)
    result = [0.0] * 48
    for v in vectors:
        for i in range(48):
            result[i] += v[i]
    return [x / n for x in result]


def pearson_r(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation coefficient."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / (n - 1)) if n > 1 else 1.0
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / (n - 1)) if n > 1 else 1.0
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)
    return cov / (sx * sy)


# ============================================================
# Data Loading
# ============================================================

def load_a1_final(a1_dir: str) -> Dict[str, List[Dict]]:
    """
    Load all *_a1_final.jsonl files.
    Returns: {atom_id: [record, ...]}
    """
    data = defaultdict(list)
    a1_path = Path(a1_dir)
    files = sorted(a1_path.glob("*_a1_final.jsonl"))
    
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                atom = rec.get("atom", "")
                if not atom:
                    continue
                # Skip failed/empty records
                if rec.get("status") == "Observation_Failed":
                    continue
                if "normalized_scores" not in rec:
                    continue
                
                data[atom].append(rec)
    
    return dict(data)


def load_lexicon_core_words(lexicon_dir: str) -> Dict[str, set]:
    """
    Load core_pool word sets for each atom.
    Returns: {atom_id: set((word_lower, pos), ...)}
    """
    core_sets = {}
    lex_path = Path(lexicon_dir)
    
    for f in sorted(lex_path.glob("*.json")):
        if f.name.startswith("_"):
            continue
        with open(f) as fh:
            entry = json.load(fh)
        
        atom_id = entry.get("atom", "")
        if not atom_id:
            continue
        if entry.get("status") == "merged":
            continue
        
        core_words = set()
        for w in entry.get("core_pool", {}).get("words", []):
            core_words.add((w["w"].lower(), w.get("pos", "?")))
        
        core_sets[atom_id] = core_words
    
    return core_sets


def load_synapse(synapse_path: str, patch_paths: List[str] = None) -> Dict[str, List[Dict]]:
    """
    Load Synapse base JSON + patches.
    Returns: {synset_id: [{"concept_id": atom_id, "raw_score": float, ...}, ...]}
    """
    with open(synapse_path) as f:
        raw = json.load(f)
    
    synapses = raw.get("synapses", raw)
    
    # Remove meta keys
    result = {}
    for sid, edges in synapses.items():
        if sid.startswith("_"):
            continue
        if isinstance(edges, list):
            result[sid] = edges
    
    # Apply patches
    tombstones = set()
    if patch_paths:
        for pp in patch_paths:
            with open(pp) as f:
                content = f.read().strip()
            
            entries = []
            # Try JSON format
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "patches" in data:
                    entries = data["patches"]
                elif isinstance(data, list):
                    entries = data
            except json.JSONDecodeError:
                # Try JSONL
                for line in content.split("\n"):
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            
            for entry in entries:
                op = entry.get("op", "add_edge")
                sid = entry.get("synset_id", "")
                atom = entry.get("atom", "")
                edge_key = entry.get("edge_key", f"{sid}::{atom}")
                
                if op == "disable_edge":
                    tombstones.add(edge_key)
                elif op == "add_edge":
                    if edge_key in tombstones:
                        continue
                    if sid not in result:
                        result[sid] = []
                    # Check for duplicate
                    found = False
                    for e in result[sid]:
                        if e.get("concept_id") == atom:
                            e["raw_score"] = entry.get("score", e.get("raw_score", 0))
                            found = True
                            break
                    if not found:
                        result[sid].append({
                            "concept_id": atom,
                            "raw_score": entry.get("score", 0.0),
                            "weight": entry.get("score", 0.0),
                            "rank": 0,
                            "patch": True,
                        })
    
    # Remove tombstoned edges
    for sid in list(result.keys()):
        result[sid] = [
            e for e in result[sid]
            if f"{sid}::{e.get('concept_id', '')}" not in tombstones
        ]
        if not result[sid]:
            del result[sid]
    
    return result


def load_dictionary(dict_path: str) -> Dict[str, Dict]:
    """Load esde_dictionary.json."""
    with open(dict_path) as f:
        raw = json.load(f)
    return raw.get("concepts", raw)


# ============================================================
# TASK 1: Atom Centroid Construction
# ============================================================

def build_centroids(a1_data: Dict[str, List[Dict]],
                    core_sets: Dict[str, set],
                    score_key: str = "normalized_scores") -> Tuple[Dict[str, List[float]], Dict]:
    """
    For each atom: centroid_48D = mean(scores of all Core words)
    
    Args:
        score_key: "normalized_scores" (softmax, sum=1) or "raw_scores" (0-10 integers)
    
    Returns: (centroids, stats)
    """
    centroids = {}
    stats = {"total_atoms": 0, "total_core_used": 0, "atoms_no_core": [],
             "score_key": score_key}
    
    for atom_id, records in sorted(a1_data.items()):
        core_set = core_sets.get(atom_id, set())
        
        # Filter to Core words only
        core_vectors = []
        for rec in records:
            word = rec.get("word", "").lower()
            pos = rec.get("pos", "?")
            
            # Check if this word is in core pool
            if (word, pos) in core_set or (word, "?") in core_set:
                scores = rec.get(score_key, {})
                if scores:
                    core_vectors.append(vec_from_scores(scores))
            else:
                # Also accept if no core set available (use all)
                pass
        
        # Fallback: if core filtering removes everything, use all words
        if not core_vectors:
            for rec in records:
                scores = rec.get(score_key, {})
                if scores:
                    core_vectors.append(vec_from_scores(scores))
            if core_vectors:
                stats["atoms_no_core"].append(atom_id)
        
        if core_vectors:
            centroids[atom_id] = vec_mean(core_vectors)
            stats["total_atoms"] += 1
            stats["total_core_used"] += len(core_vectors)
        
    return centroids, stats


# ============================================================
# TASK 2: Word-to-Atom Distance Evaluation
# ============================================================

def evaluate_word_distances(a1_data: Dict[str, List[Dict]],
                           centroids: Dict[str, List[float]],
                           score_key: str = "normalized_scores") -> List[Dict]:
    """
    For each word in A1:
    - cos_self: cosine to its assigned atom centroid
    - cos_top2: highest cosine to any OTHER atom centroid
    - margin: cos_self - cos_top2
    - l2_self: L2 distance to own centroid
    """
    results = []
    atom_ids = sorted(centroids.keys())
    
    for atom_id, records in sorted(a1_data.items()):
        if atom_id not in centroids:
            continue
        
        self_centroid = centroids[atom_id]
        
        for rec in records:
            word = rec.get("word", "")
            pos = rec.get("pos", "?")
            scores = rec.get(score_key, {})
            if not scores:
                continue
            
            word_vec = vec_from_scores(scores)
            
            # Cosine to own centroid
            cos_self = cosine_sim(word_vec, self_centroid)
            
            # Cosine to all other centroids — find top 2
            other_scores = []
            for other_atom in atom_ids:
                if other_atom == atom_id:
                    continue
                cos_other = cosine_sim(word_vec, centroids[other_atom])
                other_scores.append((other_atom, cos_other))
            
            other_scores.sort(key=lambda x: -x[1])
            cos_top2 = other_scores[0][1] if other_scores else 0.0
            top2_atom = other_scores[0][0] if other_scores else ""
            
            margin = cos_self - cos_top2
            l2_self = l2_dist(word_vec, self_centroid)
            
            results.append({
                "word": word,
                "pos": pos,
                "assigned_atom": atom_id,
                "cos_self": round(cos_self, 6),
                "cos_top2": round(cos_top2, 6),
                "top2_atom": top2_atom,
                "margin": round(margin, 6),
                "l2_self": round(l2_self, 6),
                "focus_rate": rec.get("focus_rate", 0),
                "re_observed": rec.get("re_observed", False),
            })
    
    return results


# ============================================================
# TASK 3: Synapse Edge Comparison
# ============================================================

def _try_load_wordnet():
    """Try to load NLTK WordNet. Returns wn module or None."""
    try:
        from nltk.corpus import wordnet as wn
        # Test access
        wn.synset("entity.n.01")
        return wn
    except Exception:
        return None


def _get_synset_lemmas(synset_id: str, wn) -> List[str]:
    """
    Get ALL lemma names from a WordNet synset.
    e.g., kill.v.01 → ["kill", "put to death", ...]
    """
    try:
        s = wn.synset(synset_id)
        return [l.name().lower().replace("_", " ") for l in s.lemmas()]
    except Exception:
        return []


def build_synapse_word_index(synapse: Dict[str, List[Dict]]) -> Tuple[Dict[Tuple[str, str], float], Dict[str, Any]]:
    """
    Build index: (lemma_lower, atom_id) → best_embedding_score
    
    If NLTK WordNet available: expand ALL lemmas per synset.
    Fallback: extract base lemma from synset_id only.
    
    Returns: (index, match_stats)
    """
    wn = _try_load_wordnet()
    use_nltk = wn is not None
    
    index = {}  # (lemma, atom_id) → score
    stats = {
        "mode": "nltk_full_lemma" if use_nltk else "base_lemma_only",
        "synsets_processed": 0,
        "total_lemmas_indexed": 0,
        "nltk_failures": 0,
    }
    
    for synset_id, edges in synapse.items():
        stats["synsets_processed"] += 1
        
        # Get lemmas
        lemmas = []
        if use_nltk:
            lemmas = _get_synset_lemmas(synset_id, wn)
            if not lemmas:
                stats["nltk_failures"] += 1
        
        # Fallback: extract base lemma from synset_id
        if not lemmas:
            parts = synset_id.split(".")
            if len(parts) >= 3:
                base = ".".join(parts[:-2])
            else:
                base = parts[0]
            lemmas = [base.lower().replace("_", " ")]
        
        stats["total_lemmas_indexed"] += len(lemmas)
        
        for edge in edges:
            atom_id = edge.get("concept_id", "")
            score = edge.get("raw_score", 0.0)
            
            for lemma in lemmas:
                key = (lemma, atom_id)
                if key not in index or score > index[key]:
                    index[key] = score
    
    return index, stats


def build_synapse_edge_list(synapse: Dict[str, List[Dict]]) -> List[Dict]:
    """
    Flatten Synapse into edge list for analysis.
    Returns: [{synset_id, atom_id, embedding_score}, ...]
    """
    edges = []
    for synset_id, edge_list in synapse.items():
        for edge in edge_list:
            edges.append({
                "synset_id": synset_id,
                "atom_id": edge.get("concept_id", ""),
                "embedding_score": edge.get("raw_score", 0.0),
            })
    return edges


def compare_synapse_a1(word_distances: List[Dict],
                       synapse_word_idx: Dict[Tuple[str, str], float],
                       synapse_edges: List[Dict],
                       centroids: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    Task 3: For each word in A1 that has a Synapse edge, compare.
    Also classify edges as TP/FN/FP/TN.
    """
    # Build word lookup: (word_lower, atom_id) → word distance record
    word_lookup = {}
    for wd in word_distances:
        key = (wd["word"].lower(), wd["assigned_atom"])
        # Keep best (highest cos_self)
        if key not in word_lookup or wd["cos_self"] > word_lookup[key]["cos_self"]:
            word_lookup[key] = wd
    
    # Matched comparisons (word appears in both Synapse and A1)
    matched = []
    for (lemma, atom_id), emb_score in synapse_word_idx.items():
        key = (lemma, atom_id)
        if key in word_lookup:
            wd = word_lookup[key]
            matched.append({
                "word": lemma,
                "atom_id": atom_id,
                "embedding_score": round(emb_score, 6),
                "a1_cosine": wd["cos_self"],
                "a1_margin": wd["margin"],
                "a1_l2": wd["l2_self"],
                "top2_atom": wd["top2_atom"],
                "focus_rate": wd["focus_rate"],
            })
    
    # Classification
    tp, fn, fp, tn = [], [], [], []
    
    for m in matched:
        emb_high = m["embedding_score"] >= EMBEDDING_HIGH_TH
        a1_high = m["a1_cosine"] >= A1_COSINE_HIGH_TH and m["a1_margin"] >= MARGIN_HIGH_TH
        
        if emb_high and a1_high:
            tp.append(m)
        elif not emb_high and a1_high:
            fn.append(m)
        elif emb_high and not a1_high:
            fp.append(m)
        else:
            tn.append(m)
    
    # Atom-level stats
    atom_stats = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "total": 0})
    for m in matched:
        aid = m["atom_id"]
        atom_stats[aid]["total"] += 1
        
        emb_high = m["embedding_score"] >= EMBEDDING_HIGH_TH
        a1_high = m["a1_cosine"] >= A1_COSINE_HIGH_TH and m["a1_margin"] >= MARGIN_HIGH_TH
        
        if emb_high and a1_high:
            atom_stats[aid]["tp"] += 1
        elif not emb_high and a1_high:
            atom_stats[aid]["fn"] += 1
        elif emb_high and not a1_high:
            atom_stats[aid]["fp"] += 1
        else:
            atom_stats[aid]["tn"] += 1
    
    # Unmatched Synapse edges (no A1 data)
    unmatched_synapse = 0
    for (lemma, atom_id), score in synapse_word_idx.items():
        if (lemma, atom_id) not in word_lookup:
            unmatched_synapse += 1
    
    return {
        "matched": matched,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "atom_stats": dict(atom_stats),
        "unmatched_synapse_edges": unmatched_synapse,
    }


# ============================================================
# TASK 4: Statistical Output
# ============================================================

def compute_statistics(comparison: Dict, word_distances: List[Dict],
                       centroids: Dict) -> Dict[str, Any]:
    """Compute all statistics for the report."""
    matched = comparison["matched"]
    
    # Section A: Overall correlation
    if matched:
        emb_scores = [m["embedding_score"] for m in matched]
        a1_cosines = [m["a1_cosine"] for m in matched]
        correlation = pearson_r(emb_scores, a1_cosines)
    else:
        correlation = 0.0
    
    # Cosine distribution histogram bins
    cos_bins = defaultdict(int)
    for wd in word_distances:
        bin_val = round(wd["cos_self"], 1)
        cos_bins[bin_val] += 1
    
    # Margin distribution
    margin_bins = defaultdict(int)
    for wd in word_distances:
        m = wd["margin"]
        if m < -0.1:
            margin_bins["<-0.10"] += 1
        elif m < 0.0:
            margin_bins["-0.10..0.00"] += 1
        elif m < 0.05:
            margin_bins["0.00..0.05"] += 1
        elif m < 0.10:
            margin_bins["0.05..0.10"] += 1
        elif m < 0.20:
            margin_bins["0.10..0.20"] += 1
        else:
            margin_bins[">=0.20"] += 1
    
    # Words with negative margin (assigned to wrong atom?)
    negative_margin = [wd for wd in word_distances if wd["margin"] < 0]
    negative_margin.sort(key=lambda x: x["margin"])
    
    # Atom-level precision/recall
    atom_pr = {}
    for aid, st in comparison["atom_stats"].items():
        tp = st["tp"]
        fp = st["fp"]
        fn = st["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        atom_pr[aid] = {
            "precision": precision,
            "recall": recall,
            "tp": tp, "fp": fp, "fn": fn, "tn": st["tn"],
            "total": st["total"],
        }
    
    return {
        "correlation": correlation,
        "n_matched": len(matched),
        "n_tp": len(comparison["tp"]),
        "n_fn": len(comparison["fn"]),
        "n_fp": len(comparison["fp"]),
        "n_tn": len(comparison["tn"]),
        "cos_bins": dict(cos_bins),
        "margin_bins": dict(margin_bins),
        "negative_margin_top50": negative_margin[:50],
        "atom_precision_recall": atom_pr,
        "unmatched_synapse": comparison["unmatched_synapse_edges"],
    }


# ============================================================
# Report Generation
# ============================================================

def generate_report(centroids: Dict, centroid_stats: Dict,
                    word_distances: List[Dict],
                    comparison: Dict, stats: Dict,
                    out_dir: str,
                    match_stats: Dict = None,
                    dual_mode: Dict = None):
    """Generate the final markdown report."""
    
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    lines = []
    lines.append("# ESDE Synapse v4 Comparison Report")
    lines.append(f"")
    lines.append(f"**Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**Phase**: Analysis-only (no Synapse modification)")
    lines.append(f"**Score basis**: raw_scores (0-10 integers, not softmax)")
    lines.append(f"**Thresholds**: embedding ≥ {EMBEDDING_HIGH_TH}, "
                 f"A1_cosine ≥ {A1_COSINE_HIGH_TH}, margin ≥ {MARGIN_HIGH_TH}")
    lines.append("")
    
    # Match mode info
    if match_stats:
        lines.append(f"**Lemma matching**: {match_stats['mode']}")
        lines.append(f"**Total lemmas indexed**: {match_stats['total_lemmas_indexed']} "
                     f"(from {match_stats['synsets_processed']} synsets)")
        if match_stats['nltk_failures'] > 0:
            lines.append(f"**NLTK failures**: {match_stats['nltk_failures']}")
        lines.append("")
    
    # Dual mode comparison
    if dual_mode:
        lines.append("## Score Mode Comparison (normalized vs raw)")
        lines.append("")
        lines.append("| Mode | Avg cos_self | Avg margin | Negative margin % |")
        lines.append("|------|-------------|------------|-------------------|")
        nm = dual_mode["norm"]
        rm = dual_mode["raw"]
        lines.append(f"| normalized_scores (softmax) | {nm['avg_cos']:.4f} | "
                     f"{nm['avg_margin']:.4f} | {nm['neg_count']/nm['total']*100:.1f}% |")
        lines.append(f"| **raw_scores (0-10)** | **{rm['avg_cos']:.4f}** | "
                     f"**{rm['avg_margin']:.4f}** | **{rm['neg_count']/rm['total']*100:.1f}%** |")
        lines.append("")
        lines.append("All subsequent analysis uses **raw_scores** to avoid softmax simplex compression.")
        lines.append("")
    
    # ── Section A: Overall correlation ──
    lines.append("## Section A: Overall Correlation")
    lines.append("")
    lines.append(f"- Matched word-edge pairs: **{stats['n_matched']}**")
    lines.append(f"- Pearson r (embedding_score vs A1_cosine): **{stats['correlation']:.4f}**")
    lines.append(f"- Unmatched Synapse edges (no A1 word): {stats['unmatched_synapse']}")
    lines.append("")
    lines.append("### Confusion Matrix")
    lines.append("")
    lines.append("|  | A1 High | A1 Low |")
    lines.append("|--|---------|--------|")
    lines.append(f"| **Emb High** | TP: {stats['n_tp']} | FP: {stats['n_fp']} |")
    lines.append(f"| **Emb Low**  | FN: {stats['n_fn']} | TN: {stats['n_tn']} |")
    lines.append("")
    
    total_classified = stats['n_tp'] + stats['n_fn'] + stats['n_fp'] + stats['n_tn']
    if total_classified > 0:
        accuracy = (stats['n_tp'] + stats['n_tn']) / total_classified
        lines.append(f"- Accuracy: {accuracy:.3f}")
        if (stats['n_tp'] + stats['n_fp']) > 0:
            precision = stats['n_tp'] / (stats['n_tp'] + stats['n_fp'])
            lines.append(f"- Precision (Emb High → A1 High): {precision:.3f}")
        if (stats['n_tp'] + stats['n_fn']) > 0:
            recall = stats['n_tp'] / (stats['n_tp'] + stats['n_fn'])
            lines.append(f"- Recall (A1 High captured by Emb): {recall:.3f}")
    lines.append("")
    
    # Centroid stats
    lines.append("### Centroid Construction")
    lines.append("")
    lines.append(f"- Atoms with centroids: **{centroid_stats['total_atoms']}**")
    lines.append(f"- Total Core words used: **{centroid_stats['total_core_used']}**")
    if centroid_stats["atoms_no_core"]:
        lines.append(f"- Atoms with no Core match (used all words): "
                     f"{len(centroid_stats['atoms_no_core'])}")
        for a in centroid_stats["atoms_no_core"][:10]:
            lines.append(f"  - {a}")
    lines.append("")
    
    # ── Section B: FN Candidates ──
    lines.append("## Section B: Top 50 FN Candidates")
    lines.append("")
    lines.append("These are words where embedding scored LOW but A1 measured HIGH.")
    lines.append("→ Synapse is missing these connections.")
    lines.append("")
    
    fn_sorted = sorted(comparison["fn"], key=lambda x: -x["a1_cosine"])[:50]
    if fn_sorted:
        lines.append("| # | Word | Atom | Emb Score | A1 Cosine | A1 Margin | Focus |")
        lines.append("|---|------|------|-----------|-----------|-----------|-------|")
        for i, m in enumerate(fn_sorted, 1):
            lines.append(f"| {i} | {m['word']} | {m['atom_id']} | "
                        f"{m['embedding_score']:.4f} | {m['a1_cosine']:.4f} | "
                        f"{m['a1_margin']:.4f} | {m['focus_rate']:.3f} |")
    else:
        lines.append("(No FN candidates found)")
    lines.append("")
    
    # ── Section C: FP Candidates ──
    lines.append("## Section C: Top 50 FP Candidates")
    lines.append("")
    lines.append("These are words where embedding scored HIGH but A1 measured LOW.")
    lines.append("→ Synapse may have spurious connections.")
    lines.append("")
    
    fp_sorted = sorted(comparison["fp"], key=lambda x: -x["embedding_score"])[:50]
    if fp_sorted:
        lines.append("| # | Word | Atom | Emb Score | A1 Cosine | A1 Margin | Top2 Atom | Focus |")
        lines.append("|---|------|------|-----------|-----------|-----------|-----------|-------|")
        for i, m in enumerate(fp_sorted, 1):
            lines.append(f"| {i} | {m['word']} | {m['atom_id']} | "
                        f"{m['embedding_score']:.4f} | {m['a1_cosine']:.4f} | "
                        f"{m['a1_margin']:.4f} | {m['top2_atom']} | {m['focus_rate']:.3f} |")
    else:
        lines.append("(No FP candidates found)")
    lines.append("")
    
    # ── Section D: Atom-level Statistics ──
    lines.append("## Section D: Atom-level Statistics")
    lines.append("")
    
    # Word distance summary
    if word_distances:
        avg_cos = sum(w["cos_self"] for w in word_distances) / len(word_distances)
        avg_margin = sum(w["margin"] for w in word_distances) / len(word_distances)
        negative_count = sum(1 for w in word_distances if w["margin"] < 0)
        
        lines.append("### Overall Word Distance Summary")
        lines.append("")
        lines.append(f"- Total words evaluated: **{len(word_distances)}**")
        lines.append(f"- Avg cos_self: **{avg_cos:.4f}**")
        lines.append(f"- Avg margin: **{avg_margin:.4f}**")
        lines.append(f"- Words with negative margin: **{negative_count}** "
                     f"({negative_count/len(word_distances)*100:.1f}%)")
        lines.append("")
    
    # Cosine distribution
    lines.append("### Cosine Similarity Distribution (cos_self)")
    lines.append("")
    lines.append("| Bin | Count |")
    lines.append("|-----|-------|")
    for b in sorted(stats["cos_bins"].keys()):
        lines.append(f"| {b:.1f} | {stats['cos_bins'][b]} |")
    lines.append("")
    
    # Margin distribution
    lines.append("### Margin Distribution")
    lines.append("")
    lines.append("| Range | Count |")
    lines.append("|-------|-------|")
    for label in ["<-0.10", "-0.10..0.00", "0.00..0.05", "0.05..0.10", "0.10..0.20", ">=0.20"]:
        lines.append(f"| {label} | {stats['margin_bins'].get(label, 0)} |")
    lines.append("")
    
    # Worst negative margin words
    if stats["negative_margin_top50"]:
        lines.append("### Top 50 Negative Margin Words (potential misassignments)")
        lines.append("")
        lines.append("| # | Word | Assigned | cos_self | Top2 Atom | cos_top2 | Margin |")
        lines.append("|---|------|----------|----------|-----------|----------|--------|")
        for i, wd in enumerate(stats["negative_margin_top50"][:50], 1):
            lines.append(f"| {i} | {wd['word']} | {wd['assigned_atom']} | "
                        f"{wd['cos_self']:.4f} | {wd['top2_atom']} | "
                        f"{wd['cos_top2']:.4f} | {wd['margin']:.4f} |")
        lines.append("")
    
    # Atom precision/recall table (only atoms with data)
    atom_pr = stats["atom_precision_recall"]
    if atom_pr:
        lines.append("### Atom-level Precision/Recall")
        lines.append("")
        lines.append("| Atom | TP | FP | FN | TN | Precision | Recall | Total |")
        lines.append("|------|----|----|----|----|-----------|--------|-------|")
        for aid in sorted(atom_pr.keys()):
            pr = atom_pr[aid]
            p_str = f"{pr['precision']:.3f}" if pr['precision'] is not None else "N/A"
            r_str = f"{pr['recall']:.3f}" if pr['recall'] is not None else "N/A"
            lines.append(f"| {aid} | {pr['tp']} | {pr['fp']} | {pr['fn']} | "
                        f"{pr['tn']} | {p_str} | {r_str} | {pr['total']} |")
        lines.append("")
    
    # ── Section E: Preliminary Structural Verdict ──
    lines.append("## Section E: Preliminary Structural Verdict")
    lines.append("")
    
    # Generate verdict
    corr = stats["correlation"]
    n_fn = stats["n_fn"]
    n_fp = stats["n_fp"]
    n_matched = stats["n_matched"]
    
    verdict_lines = []
    verdict_lines.append(f"1. Embedding-A1 correlation: r={corr:.4f} — "
                        f"{'weak' if abs(corr) < 0.3 else 'moderate' if abs(corr) < 0.6 else 'strong'} "
                        f"linear relationship.")
    
    if n_matched > 0:
        fn_rate = n_fn / n_matched * 100
        fp_rate = n_fp / n_matched * 100
        verdict_lines.append(f"2. FN rate: {fn_rate:.1f}% — embedding misses these valid connections.")
        verdict_lines.append(f"3. FP rate: {fp_rate:.1f}% — embedding assigns these spuriously.")
    
    if word_distances:
        neg_margin_rate = sum(1 for w in word_distances if w["margin"] < 0) / len(word_distances) * 100
        verdict_lines.append(f"4. Negative-margin words: {neg_margin_rate:.1f}% — "
                           f"words closer to another atom than their assigned one.")
    
    verdict_lines.append(f"5. A1 empirical data available for Synapse v4 evidence-based reconstruction.")
    
    for vl in verdict_lines:
        lines.append(vl)
    lines.append("")
    
    # Write report
    report_path = out / "synapse_v4_comparison.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"\n  ✅ Report: {report_path}")
    return report_path


def save_csv_outputs(word_distances: List[Dict], comparison: Dict,
                     centroids: Dict, out_dir: str,
                     centroids_norm: Dict = None):
    """Save detailed CSV outputs."""
    out = Path(out_dir)
    
    # Word distances CSV
    wd_path = out / "word_distances.csv"
    with open(wd_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "word", "pos", "assigned_atom", "cos_self", "cos_top2",
            "top2_atom", "margin", "l2_self", "focus_rate", "re_observed"
        ])
        w.writeheader()
        for wd in sorted(word_distances, key=lambda x: x["margin"]):
            w.writerow(wd)
    print(f"  ✅ Word distances: {wd_path} ({len(word_distances)} rows)")
    
    # Matched comparisons CSV
    if comparison["matched"]:
        mc_path = out / "synapse_a1_matched.csv"
        with open(mc_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "word", "atom_id", "embedding_score", "a1_cosine",
                "a1_margin", "a1_l2", "top2_atom", "focus_rate"
            ])
            w.writeheader()
            for m in sorted(comparison["matched"], key=lambda x: -x["a1_margin"]):
                w.writerow(m)
        print(f"  ✅ Matched comparisons: {mc_path} ({len(comparison['matched'])} rows)")
    
    # Raw centroid matrix CSV
    cent_path = out / "atom_centroids_48d_raw.csv"
    with open(cent_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["atom_id"] + SLOT_IDS)
        for aid in sorted(centroids.keys()):
            w.writerow([aid] + [f"{v:.6f}" for v in centroids[aid]])
    print(f"  ✅ Centroid matrix (raw): {cent_path} ({len(centroids)} atoms × 48 dims)")
    
    # Normalized centroid matrix CSV
    if centroids_norm:
        cent_norm_path = out / "atom_centroids_48d_normalized.csv"
        with open(cent_norm_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["atom_id"] + SLOT_IDS)
            for aid in sorted(centroids_norm.keys()):
                w.writerow([aid] + [f"{v:.6f}" for v in centroids_norm[aid]])
        print(f"  ✅ Centroid matrix (normalized): {cent_norm_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="ESDE Synapse v4 Comparison Experiment"
    )
    parser.add_argument("--a1-dir", required=True,
                        help="Directory containing *_a1_final.jsonl files")
    parser.add_argument("--lexicon-dir", required=True,
                        help="Directory containing lexicon entry JSONs")
    parser.add_argument("--synapse", required=True,
                        help="Base Synapse JSON (esde_synapses_v3.json)")
    parser.add_argument("--patches", nargs="*", default=[],
                        help="Synapse patch files (applied in order)")
    parser.add_argument("--dictionary", required=True,
                        help="esde_dictionary.json")
    parser.add_argument("--out-dir", default="synapse_v4_report",
                        help="Output directory")
    args = parser.parse_args()
    
    print("=" * 70)
    print("  ESDE Synapse v4 Comparison Experiment")
    print("  Analysis-only — no Synapse modification")
    print("=" * 70)
    
    # ── Load Data ──
    print("\n[1/6] Loading A1 final data...")
    a1_data = load_a1_final(args.a1_dir)
    total_records = sum(len(recs) for recs in a1_data.values())
    print(f"  Atoms: {len(a1_data)}, Records: {total_records}")
    
    print("\n[2/6] Loading Lexicon core sets...")
    core_sets = load_lexicon_core_words(args.lexicon_dir)
    total_core = sum(len(s) for s in core_sets.values())
    print(f"  Atoms with core: {len(core_sets)}, Total core words: {total_core}")
    
    print("\n[3/6] Loading Synapse (base + patches)...")
    synapse = load_synapse(args.synapse, args.patches)
    total_edges = sum(len(edges) for edges in synapse.values())
    print(f"  Synsets: {len(synapse)}, Edges: {total_edges}")
    
    print("\n[4/6] Loading dictionary...")
    dictionary = load_dictionary(args.dictionary)
    print(f"  Atoms in dictionary: {len(dictionary)}")
    
    # ── Task 1: Centroids ──
    print("\n" + "=" * 70)
    print("  TASK 1: Atom Centroid Construction")
    print("=" * 70)
    
    # Normalized (softmax) centroids
    centroids_norm, cstats_norm = build_centroids(a1_data, core_sets, "normalized_scores")
    print(f"  [normalized_scores] Centroids: {cstats_norm['total_atoms']}, "
          f"Core words: {cstats_norm['total_core_used']}")
    
    # Raw (0-10) centroids
    centroids_raw, cstats_raw = build_centroids(a1_data, core_sets, "raw_scores")
    print(f"  [raw_scores]        Centroids: {cstats_raw['total_atoms']}, "
          f"Core words: {cstats_raw['total_core_used']}")
    
    if cstats_norm["atoms_no_core"]:
        print(f"  ⚠ No core match for {len(cstats_norm['atoms_no_core'])} atoms "
              f"(used all words as fallback)")
    
    # ── Task 2: Word Distances (both modes) ──
    print("\n" + "=" * 70)
    print("  TASK 2: Word-to-Atom Distance Evaluation")
    print("=" * 70)
    
    # Normalized mode
    word_distances_norm = evaluate_word_distances(a1_data, centroids_norm,
                                                  score_key="normalized_scores")
    neg_norm = sum(1 for w in word_distances_norm if w["margin"] < 0)
    avg_cos_norm = sum(w["cos_self"] for w in word_distances_norm) / len(word_distances_norm) if word_distances_norm else 0
    avg_margin_norm = sum(w["margin"] for w in word_distances_norm) / len(word_distances_norm) if word_distances_norm else 0
    
    # Raw mode
    word_distances_raw = evaluate_word_distances(a1_data, centroids_raw,
                                                 score_key="raw_scores")
    neg_raw = sum(1 for w in word_distances_raw if w["margin"] < 0)
    avg_cos_raw = sum(w["cos_self"] for w in word_distances_raw) / len(word_distances_raw) if word_distances_raw else 0
    avg_margin_raw = sum(w["margin"] for w in word_distances_raw) / len(word_distances_raw) if word_distances_raw else 0
    
    print(f"  {'Mode':<20s} {'Words':>7s} {'Avg cos_self':>13s} {'Avg margin':>12s} {'Neg margin':>12s}")
    print(f"  {'-'*20} {'-'*7} {'-'*13} {'-'*12} {'-'*12}")
    print(f"  {'normalized_scores':<20s} {len(word_distances_norm):>7d} {avg_cos_norm:>13.4f} "
          f"{avg_margin_norm:>12.4f} {neg_norm:>7d} ({neg_norm/len(word_distances_norm)*100:.1f}%)")
    print(f"  {'raw_scores':<20s} {len(word_distances_raw):>7d} {avg_cos_raw:>13.4f} "
          f"{avg_margin_raw:>12.4f} {neg_raw:>7d} ({neg_raw/len(word_distances_raw)*100:.1f}%)")
    
    # Use raw_scores for downstream analysis (better separation expected)
    word_distances = word_distances_raw
    centroids = centroids_raw
    centroid_stats = cstats_raw
    print(f"\n  → Using raw_scores for Task 3/4 (avoids softmax simplex compression)")
    
    # ── Task 3: Synapse Comparison ──
    print("\n" + "=" * 70)
    print("  TASK 3: Synapse Edge Comparison")
    print("=" * 70)
    synapse_word_idx, match_stats = build_synapse_word_index(synapse)
    synapse_edges = build_synapse_edge_list(synapse)
    print(f"  Matching mode: {match_stats['mode']}")
    print(f"  Synsets processed: {match_stats['synsets_processed']}")
    print(f"  Total lemmas indexed: {match_stats['total_lemmas_indexed']}")
    if match_stats['nltk_failures'] > 0:
        print(f"  ⚠ NLTK lookup failures: {match_stats['nltk_failures']}")
    print(f"  Synapse word-atom pairs: {len(synapse_word_idx)}")
    print(f"  Synapse edges total: {len(synapse_edges)}")
    
    comparison = compare_synapse_a1(word_distances, synapse_word_idx,
                                    synapse_edges, centroids)
    print(f"  Matched pairs: {len(comparison['matched'])}")
    print(f"  TP: {len(comparison['tp'])}, FN: {len(comparison['fn'])}, "
          f"FP: {len(comparison['fp'])}, TN: {len(comparison['tn'])}")
    print(f"  Unmatched Synapse edges: {comparison['unmatched_synapse_edges']}")
    
    # ── Task 4: Statistics ──
    print("\n" + "=" * 70)
    print("  TASK 4: Statistical Output")
    print("=" * 70)
    stats = compute_statistics(comparison, word_distances, centroids)
    print(f"  Correlation (embedding vs A1): r = {stats['correlation']:.4f}")
    
    # ── Generate Report ──
    print("\n" + "=" * 70)
    print("  Generating Report")
    print("=" * 70)
    
    report_path = generate_report(centroids, centroid_stats, word_distances,
                                  comparison, stats, args.out_dir,
                                  match_stats=match_stats,
                                  dual_mode={
                                      "norm": {"avg_cos": avg_cos_norm, "avg_margin": avg_margin_norm,
                                               "neg_count": neg_norm, "total": len(word_distances_norm)},
                                      "raw": {"avg_cos": avg_cos_raw, "avg_margin": avg_margin_raw,
                                              "neg_count": neg_raw, "total": len(word_distances_raw)},
                                  })
    save_csv_outputs(word_distances, comparison, centroids, args.out_dir,
                     centroids_norm=centroids_norm)
    
    print("\n" + "=" * 70)
    print("  COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()