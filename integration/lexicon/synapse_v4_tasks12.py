#!/usr/bin/env python3
"""
ESDE Synapse v4 — Task 1 + Task 2
===================================
Task 1: Local Margin Recalculation (Neighborhood-Aware)
Task 2: Synapse v4 Candidate Generation (A1-Based)

Per GPT directive (2026-03-01). Analysis-only. No Synapse modification.

Usage:
  python3 synapse_v4_tasks12.py \
    --centroids synapse_v4_report/atom_centroids_48d_raw.csv \
    --word-distances synapse_v4_report/word_distances.csv \
    --a1-dir audit_output/ \
    --dictionary esde_dictionary.json \
    --constitution proposals.json \
    --out-dir synapse_v4_report/
"""

import json
import math
import csv
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any, Optional
from datetime import datetime, timezone


# ============================================================
# 48 Slot IDs (canonical order)
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


# ============================================================
# Math Utilities
# ============================================================

def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


def vec_from_scores(scores: Dict[str, float]) -> List[float]:
    return [scores.get(s, 0.0) for s in SLOT_IDS]


# ============================================================
# Data Loading
# ============================================================

def load_centroids(path: str) -> Dict[str, List[float]]:
    """Load atom centroid matrix from CSV."""
    centroids = {}
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)  # atom_id, slot1, slot2, ...
        for row in reader:
            aid = row[0]
            vec = [float(v) for v in row[1:]]
            centroids[aid] = vec
    return centroids


def load_word_distances(path: str) -> List[Dict]:
    """Load word_distances.csv."""
    results = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "word": row["word"],
                "pos": row["pos"],
                "assigned_atom": row["assigned_atom"],
                "cos_self": float(row["cos_self"]),
                "cos_top2": float(row["cos_top2"]),
                "top2_atom": row["top2_atom"],
                "margin": float(row["margin"]),
                "l2_self": float(row["l2_self"]),
                "focus_rate": float(row["focus_rate"]),
                "re_observed": row["re_observed"],
            })
    return results


def load_dictionary(path: str) -> Dict[str, Dict]:
    with open(path) as f:
        raw = json.load(f)
    return raw.get("concepts", raw)


def load_constitution(path: str) -> List[Dict]:
    """Load proposals.json (Constitution v1.0)."""
    if not Path(path).exists():
        return []
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("proposals", [])


def load_a1_final(a1_dir: str) -> Dict[str, List[Dict]]:
    """Load all *_a1_final.jsonl files. Returns: {atom_id: [record, ...]}"""
    data = defaultdict(list)
    for f in sorted(Path(a1_dir).glob("*_a1_final.jsonl")):
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
                if not atom or rec.get("status") == "Observation_Failed":
                    continue
                if "raw_scores" not in rec:
                    continue
                data[atom].append(rec)
    return dict(data)


# ============================================================
# TASK 1: Neighborhood Construction
# ============================================================

def build_neighborhoods(centroids: Dict[str, List[float]],
                        dictionary: Dict[str, Dict],
                        constitution: List[Dict],
                        top_k: int = 20) -> Dict[str, Set[str]]:
    """
    For each atom, construct neighborhood N(atom):
    1. Same category atoms
    2. Explicit symmetric pair
    3. Couple candidates (Constitution)
    4. Top-20 nearest by centroid cosine
    
    Returns: {atom_id: set of neighbor atom_ids (excluding self)}
    """
    atom_ids = sorted(centroids.keys())
    
    # Category index
    cat_map = {}
    for aid in atom_ids:
        cat = aid.split(".")[0] if "." in aid else ""
        if aid in dictionary:
            cat = dictionary[aid].get("category", cat)
        cat_map[aid] = cat
    
    cat_members = defaultdict(set)
    for aid, cat in cat_map.items():
        cat_members[cat].add(aid)
    
    # Symmetric pairs
    sym_pairs = {}
    for aid, entry in dictionary.items():
        sp = entry.get("symmetric_pair", "")
        if sp and sp in centroids:
            sym_pairs[aid] = sp
    
    # Couple links from Constitution
    couple_links = defaultdict(set)
    for prop in constitution:
        pattern = prop.get("pattern", "")
        if "COUPLE" in pattern.upper() or pattern in ("B_COUPLE", "C_COUPLE"):
            a = prop.get("atom_a", "")
            b = prop.get("atom_b", "")
            if a in centroids and b in centroids:
                couple_links[a].add(b)
                couple_links[b].add(a)
    
    # Precompute all pairwise cosines for top-k
    # (O(n^2) but n=325, so ~53k pairs — fast enough)
    all_cosines = {}
    for i, a in enumerate(atom_ids):
        for j, b in enumerate(atom_ids):
            if i >= j:
                continue
            c = cosine_sim(centroids[a], centroids[b])
            all_cosines[(a, b)] = c
            all_cosines[(b, a)] = c
    
    # Build neighborhoods
    neighborhoods = {}
    stats = {"avg_size": 0, "min_size": 999, "max_size": 0}
    
    for aid in atom_ids:
        neighbors = set()
        
        # 1. Same category
        cat = cat_map.get(aid, "")
        neighbors.update(cat_members.get(cat, set()))
        
        # 2. Symmetric pair
        if aid in sym_pairs:
            neighbors.add(sym_pairs[aid])
        
        # 3. Couple links
        neighbors.update(couple_links.get(aid, set()))
        
        # 4. Top-k nearest by cosine
        dists = []
        for other in atom_ids:
            if other == aid:
                continue
            c = all_cosines.get((aid, other), 0.0)
            dists.append((other, c))
        dists.sort(key=lambda x: -x[1])
        for other, _ in dists[:top_k]:
            neighbors.add(other)
        
        # Remove self
        neighbors.discard(aid)
        
        neighborhoods[aid] = neighbors
        stats["avg_size"] += len(neighbors)
        stats["min_size"] = min(stats["min_size"], len(neighbors))
        stats["max_size"] = max(stats["max_size"], len(neighbors))
    
    stats["avg_size"] = stats["avg_size"] / len(atom_ids) if atom_ids else 0
    
    return neighborhoods, stats


# ============================================================
# TASK 1: Local Margin Calculation
# ============================================================

def compute_local_margins(word_distances: List[Dict],
                          centroids: Dict[str, List[float]],
                          neighborhoods: Dict[str, Set[str]],
                          a1_data: Dict[str, List[Dict]]) -> List[Dict]:
    """
    For each word:
    margin_local = cos_self - max(cos(word, centroid_a)) for a in N(atom) excluding self
    
    Uses raw_scores from A1 data for word vectors.
    """
    # Build word vector lookup from A1 data (raw_scores)
    word_vectors = {}  # (word_lower, atom_id) → vec
    for atom_id, records in a1_data.items():
        for rec in records:
            w = rec.get("word", "").lower()
            rs = rec.get("raw_scores", {})
            if rs:
                word_vectors[(w, atom_id)] = vec_from_scores(rs)
    
    results = []
    
    for wd in word_distances:
        word = wd["word"].lower()
        atom = wd["assigned_atom"]
        
        # Get word vector
        wvec = word_vectors.get((word, atom))
        if wvec is None:
            results.append({**wd, "margin_local": None, "local_top_atom": ""})
            continue
        
        # Get neighborhood
        nhood = neighborhoods.get(atom, set())
        if not nhood:
            results.append({**wd, "margin_local": wd["margin"],
                          "local_top_atom": wd["top2_atom"]})
            continue
        
        # Compute cosine to all neighbors
        best_cos = -1.0
        best_atom = ""
        for neighbor in nhood:
            if neighbor not in centroids:
                continue
            c = cosine_sim(wvec, centroids[neighbor])
            if c > best_cos:
                best_cos = c
                best_atom = neighbor
        
        cos_self = wd["cos_self"]
        margin_local = cos_self - best_cos if best_cos > -0.5 else cos_self
        
        results.append({
            **wd,
            "margin_local": round(margin_local, 6),
            "local_top_atom": best_atom,
        })
    
    return results


def analyze_local_margins(local_results: List[Dict]) -> Dict[str, Any]:
    """Compute summary statistics for local margin analysis."""
    valid = [r for r in local_results if r["margin_local"] is not None]
    
    if not valid:
        return {"error": "No valid local margins computed"}
    
    neg_global = sum(1 for r in valid if r["margin"] < 0)
    neg_local = sum(1 for r in valid if r["margin_local"] < 0)
    
    avg_margin_global = sum(r["margin"] for r in valid) / len(valid)
    avg_margin_local = sum(r["margin_local"] for r in valid) / len(valid)
    
    # Per-atom breakdown
    atom_stats = defaultdict(lambda: {"total": 0, "neg_local": 0, "neg_global": 0,
                                       "sum_local": 0.0, "sum_global": 0.0})
    for r in valid:
        a = r["assigned_atom"]
        atom_stats[a]["total"] += 1
        atom_stats[a]["sum_local"] += r["margin_local"]
        atom_stats[a]["sum_global"] += r["margin"]
        if r["margin_local"] < 0:
            atom_stats[a]["neg_local"] += 1
        if r["margin"] < 0:
            atom_stats[a]["neg_global"] += 1
    
    # Atoms where local negative margin > 50%
    problem_atoms = []
    for aid, st in sorted(atom_stats.items()):
        if st["total"] > 0:
            neg_pct = st["neg_local"] / st["total"] * 100
            if neg_pct > 50:
                problem_atoms.append({
                    "atom": aid,
                    "total": st["total"],
                    "neg_local": st["neg_local"],
                    "neg_local_pct": round(neg_pct, 1),
                    "avg_margin_local": round(st["sum_local"] / st["total"], 4),
                })
    
    problem_atoms.sort(key=lambda x: -x["neg_local_pct"])
    
    # Margin_local distribution
    margin_bins = defaultdict(int)
    for r in valid:
        m = r["margin_local"]
        if m < -0.20:
            margin_bins["<-0.20"] += 1
        elif m < -0.10:
            margin_bins["-0.20..-0.10"] += 1
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
    
    # Words that flip from negative (global) to positive (local)
    flipped = sum(1 for r in valid if r["margin"] < 0 and r["margin_local"] >= 0)
    
    return {
        "total_words": len(valid),
        "neg_global": neg_global,
        "neg_global_pct": round(neg_global / len(valid) * 100, 1),
        "neg_local": neg_local,
        "neg_local_pct": round(neg_local / len(valid) * 100, 1),
        "avg_margin_global": round(avg_margin_global, 4),
        "avg_margin_local": round(avg_margin_local, 4),
        "flipped_to_positive": flipped,
        "flipped_pct": round(flipped / len(valid) * 100, 1),
        "problem_atoms_gt50": problem_atoms,
        "margin_bins": dict(margin_bins),
        "atom_stats": {aid: dict(st) for aid, st in atom_stats.items()},
    }


# ============================================================
# TASK 2: Synapse v4 Candidate Generation
# ============================================================

def generate_v4_candidates(a1_data: Dict[str, List[Dict]],
                           centroids: Dict[str, List[float]],
                           neighborhoods: Dict[str, Set[str]] = None,
                           top_k: int = 5) -> List[Dict]:
    """
    For each word in A1:
    - Compute cosine to all atom centroids (using raw_scores)
    - Select top_k atoms
    - Do NOT collapse to single winner
    
    Returns: list of candidate records
    """
    atom_ids = sorted(centroids.keys())
    candidates = []
    
    for atom_id, records in sorted(a1_data.items()):
        for rec in records:
            word = rec.get("word", "")
            pos = rec.get("pos", "?")
            rs = rec.get("raw_scores", {})
            if not rs:
                continue
            
            word_vec = vec_from_scores(rs)
            
            # Compute cosine to all centroids
            scores = []
            for aid in atom_ids:
                c = cosine_sim(word_vec, centroids[aid])
                scores.append((aid, c))
            
            scores.sort(key=lambda x: -x[1])
            
            # Also compute with normalized_scores if available
            ns = rec.get("normalized_scores", {})
            norm_vec = vec_from_scores(ns) if ns else None
            
            top_atoms = []
            for rank, (aid, cos_raw) in enumerate(scores[:top_k], 1):
                entry = {
                    "atom": aid,
                    "cos_raw": round(cos_raw, 6),
                    "rank": rank,
                }
                
                # Normalized cosine
                if norm_vec and aid in centroids:
                    # We'd need normalized centroids too; skip for now
                    entry["cos_norm"] = None
                
                # Local margin (if neighborhood available)
                if neighborhoods and atom_id in neighborhoods:
                    nhood = neighborhoods[atom_id]
                    local_neighbors = [s for a, s in scores if a in nhood and a != atom_id]
                    if local_neighbors:
                        entry["margin_local"] = round(cos_raw - max(local_neighbors), 6) if rank == 1 else None
                
                top_atoms.append(entry)
            
            candidates.append({
                "word": word,
                "pos": pos,
                "assigned_atom": atom_id,
                "top_k_atoms": top_atoms,
                "self_rank": next((i+1 for i, (a, _) in enumerate(scores) if a == atom_id), -1),
                "cos_self_raw": round(
                    next((c for a, c in scores if a == atom_id), 0.0), 6
                ),
                "focus_rate": rec.get("focus_rate", 0),
                "status": "analysis_only",
            })
    
    return candidates


# ============================================================
# Report Generation
# ============================================================

def write_task1_report(analysis: Dict, out_dir: str):
    """Write local_margin_summary.md"""
    out = Path(out_dir)
    lines = []
    
    lines.append("# ESDE Synapse v4 — Task 1: Local Margin Analysis")
    lines.append(f"")
    lines.append(f"**Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**Phase**: Analysis-only (no Synapse modification)")
    lines.append(f"**Score basis**: raw_scores (0-10 integers)")
    lines.append("")
    
    lines.append("## Global vs Local Margin Comparison")
    lines.append("")
    lines.append("| Metric | Global | Local (Neighborhood) |")
    lines.append("|--------|--------|---------------------|")
    lines.append(f"| Negative margin % | **{analysis['neg_global_pct']}%** | "
                 f"**{analysis['neg_local_pct']}%** |")
    lines.append(f"| Avg margin | {analysis['avg_margin_global']} | "
                 f"{analysis['avg_margin_local']} |")
    lines.append(f"| Total words | {analysis['total_words']} | {analysis['total_words']} |")
    lines.append("")
    lines.append(f"**Flipped to positive** (negative global → positive local): "
                 f"**{analysis['flipped_to_positive']}** ({analysis['flipped_pct']}%)")
    lines.append("")
    
    # Interpretation
    reduction = analysis['neg_global_pct'] - analysis['neg_local_pct']
    lines.append("## Interpretation")
    lines.append("")
    lines.append(f"Global negative margin: {analysis['neg_global_pct']}%")
    lines.append(f"Local negative margin:  {analysis['neg_local_pct']}%")
    lines.append(f"Reduction:              {reduction:.1f} percentage points")
    lines.append("")
    
    if analysis['neg_local_pct'] < 50:
        lines.append("**Result**: Most words are closer to their own atom than to "
                     "neighborhood atoms. The high global negative margin is primarily "
                     "an artifact of distant atoms being geometrically closer in 48D space, "
                     "not a structural assignment problem.")
    elif analysis['neg_local_pct'] < 70:
        lines.append("**Result**: Moderate local negative margin suggests some atoms have "
                     "genuinely overlapping coordinate profiles with neighbors. "
                     "These atoms may benefit from Constitution couple/merge review.")
    else:
        lines.append("**Result**: High local negative margin indicates substantial overlap "
                     "between neighboring atoms in 48D space. Investigate problem atoms below.")
    lines.append("")
    
    # Margin distribution
    lines.append("## Local Margin Distribution")
    lines.append("")
    lines.append("| Range | Count |")
    lines.append("|-------|-------|")
    for label in ["<-0.20", "-0.20..-0.10", "-0.10..0.00", "0.00..0.05",
                  "0.05..0.10", "0.10..0.20", ">=0.20"]:
        lines.append(f"| {label} | {analysis['margin_bins'].get(label, 0)} |")
    lines.append("")
    
    # Problem atoms
    problems = analysis["problem_atoms_gt50"]
    lines.append(f"## Problem Atoms (local negative margin > 50%): {len(problems)}")
    lines.append("")
    if problems:
        lines.append("| Atom | Total Words | Neg Local | Neg % | Avg Margin Local |")
        lines.append("|------|-------------|-----------|-------|------------------|")
        for p in problems[:50]:
            lines.append(f"| {p['atom']} | {p['total']} | {p['neg_local']} | "
                        f"{p['neg_local_pct']}% | {p['avg_margin_local']} |")
    else:
        lines.append("(None)")
    lines.append("")
    
    report_path = out / "local_margin_summary.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"  ✅ Report: {report_path}")
    return report_path


def write_task1_csv(local_results: List[Dict], out_dir: str):
    """Write local_margin_analysis.csv"""
    out = Path(out_dir)
    csv_path = out / "local_margin_analysis.csv"
    
    fieldnames = ["word", "pos", "assigned_atom", "cos_self", "cos_top2",
                  "top2_atom", "margin", "margin_local", "local_top_atom",
                  "l2_self", "focus_rate"]
    
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in sorted(local_results, key=lambda x: x.get("margin_local") or 999):
            w.writerow(r)
    
    print(f"  ✅ CSV: {csv_path} ({len(local_results)} rows)")


def write_task2_output(candidates: List[Dict], out_dir: str):
    """Write synapse_v4_candidates.jsonl"""
    out = Path(out_dir)
    jsonl_path = out / "synapse_v4_candidates.jsonl"
    
    with open(jsonl_path, "w") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    
    print(f"  ✅ Candidates: {jsonl_path} ({len(candidates)} records)")
    
    # Summary stats
    self_rank_dist = defaultdict(int)
    for c in candidates:
        sr = c["self_rank"]
        if sr <= 5:
            self_rank_dist[f"rank_{sr}"] += 1
        else:
            self_rank_dist["rank_6+"] += 1
    
    print(f"  Self-rank distribution (where does assigned atom appear in top-k):")
    for k in sorted(self_rank_dist.keys()):
        pct = self_rank_dist[k] / len(candidates) * 100
        print(f"    {k}: {self_rank_dist[k]} ({pct:.1f}%)")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="ESDE Synapse v4 — Task 1 (Local Margin) + Task 2 (Candidates)"
    )
    parser.add_argument("--centroids", required=True,
                        help="atom_centroids_48d_raw.csv from synapse_v4_compare.py")
    parser.add_argument("--word-distances", required=True,
                        help="word_distances.csv from synapse_v4_compare.py")
    parser.add_argument("--a1-dir", required=True,
                        help="Directory containing *_a1_final.jsonl")
    parser.add_argument("--dictionary", required=True,
                        help="esde_dictionary.json")
    parser.add_argument("--constitution", default="",
                        help="proposals.json (optional, for couple links)")
    parser.add_argument("--top-k-neighbors", type=int, default=20,
                        help="Top-K nearest atoms for neighborhood (default: 20)")
    parser.add_argument("--top-k-candidates", type=int, default=5,
                        help="Top-K atoms per word for v4 candidates (default: 5)")
    parser.add_argument("--out-dir", default="synapse_v4_report",
                        help="Output directory")
    args = parser.parse_args()
    
    print("=" * 70)
    print("  ESDE Synapse v4 — Task 1 + Task 2")
    print("  Analysis-only — no Synapse modification")
    print("=" * 70)
    
    # ── Load Data ──
    print("\n[1/5] Loading centroids...")
    centroids = load_centroids(args.centroids)
    print(f"  Atoms: {len(centroids)}")
    
    print("\n[2/5] Loading word distances...")
    word_distances = load_word_distances(args.word_distances)
    print(f"  Words: {len(word_distances)}")
    
    print("\n[3/5] Loading A1 final data...")
    a1_data = load_a1_final(args.a1_dir)
    total_records = sum(len(recs) for recs in a1_data.values())
    print(f"  Atoms: {len(a1_data)}, Records: {total_records}")
    
    print("\n[4/5] Loading dictionary...")
    dictionary = load_dictionary(args.dictionary)
    print(f"  Atoms: {len(dictionary)}")
    
    constitution = []
    if args.constitution and Path(args.constitution).exists():
        print(f"\n[4b/5] Loading constitution...")
        constitution = load_constitution(args.constitution)
        couples = [p for p in constitution if "COUPLE" in p.get("pattern", "").upper()]
        print(f"  Proposals: {len(constitution)} (couples: {len(couples)})")
    else:
        print(f"\n[4b/5] No constitution file (couple links skipped)")
    
    # ── Task 1: Neighborhoods + Local Margin ──
    print("\n" + "=" * 70)
    print(f"  TASK 1: Local Margin (top-{args.top_k_neighbors} neighborhood)")
    print("=" * 70)
    
    print("\n  Building neighborhoods...")
    neighborhoods, nhood_stats = build_neighborhoods(
        centroids, dictionary, constitution, top_k=args.top_k_neighbors
    )
    print(f"  Avg neighborhood size: {nhood_stats['avg_size']:.1f}")
    print(f"  Range: [{nhood_stats['min_size']}, {nhood_stats['max_size']}]")
    
    print("\n  Computing local margins...")
    local_results = compute_local_margins(word_distances, centroids,
                                          neighborhoods, a1_data)
    
    analysis = analyze_local_margins(local_results)
    
    print(f"\n  {'Metric':<25s} {'Global':>10s} {'Local':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")
    print(f"  {'Negative margin %':<25s} {analysis['neg_global_pct']:>9.1f}% "
          f"{analysis['neg_local_pct']:>9.1f}%")
    print(f"  {'Avg margin':<25s} {analysis['avg_margin_global']:>10.4f} "
          f"{analysis['avg_margin_local']:>10.4f}")
    print(f"  {'Flipped to positive':<25s} {'':>10s} "
          f"{analysis['flipped_to_positive']:>7d} ({analysis['flipped_pct']:.1f}%)")
    print(f"  {'Problem atoms (>50%)':<25s} {'':>10s} "
          f"{len(analysis['problem_atoms_gt50']):>10d}")
    
    # Write Task 1 outputs
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    write_task1_report(analysis, args.out_dir)
    write_task1_csv(local_results, args.out_dir)
    
    # ── Task 2: Candidate Generation ──
    print("\n" + "=" * 70)
    print(f"  TASK 2: Synapse v4 Candidate Generation (top-{args.top_k_candidates})")
    print("=" * 70)
    
    candidates = generate_v4_candidates(a1_data, centroids,
                                         neighborhoods=neighborhoods,
                                         top_k=args.top_k_candidates)
    print(f"  Candidates generated: {len(candidates)}")
    
    write_task2_output(candidates, args.out_dir)
    
    print("\n" + "=" * 70)
    print("  COMPLETE — Task 1 + Task 2")
    print("=" * 70)


if __name__ == "__main__":
    main()
