#!/usr/bin/env python3
"""
ESDE Synapse v4 — Task 3: ObsC Grounding Evaluation (v3 vs v4 Candidates)
==========================================================================
Compares grounding quality between:
  - v3: Current Synapse (embedding-based)
  - v4: A1-derived candidates (centroid cosine-based)

Method: Re-ground existing SVO edges using v4 candidate lookup,
then compare against v3 grounding results.

DOES NOT re-run spaCy or the full relation pipeline.
Uses existing v3 edge JSONL files as the SVO source.

Per GPT directive (2026-03-01). Analysis-only. No permanent changes.

Usage:
  python3 synapse_v4_task3.py \
    --v3-edges-dir obsC_output/ \
    --v4-candidates synapse_v4_report/synapse_v4_candidates.jsonl \
    --dictionary esde_dictionary.json \
    --out-dir synapse_v4_report/
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timezone


# Light verb stoplist (from relation_logger.py)
LIGHT_VERB_STOPLIST = {
    "be", "have", "do", "make", "get", "go", "come", "take",
    "give", "say", "know", "see",
}

# POS Guard blocked categories (from relation_logger.py)
POS_GUARD_BLOCKED = {"NAT", "MAT", "PRP", "SPA"}


# ============================================================
# Data Loading
# ============================================================

def load_v3_edges(edges_dir: str) -> List[Dict]:
    """Load all *_edges.jsonl files from ObsC output."""
    edges = []
    for f in sorted(Path(edges_dir).glob("*_edges.jsonl")):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    edge = json.loads(line)
                    edges.append(edge)
                except json.JSONDecodeError:
                    continue
    return edges


def load_v4_candidates(path: str) -> Dict[str, List[Dict]]:
    """
    Load synapse_v4_candidates.jsonl.
    Build lookup: word_lower → [{atom, cos_raw, rank}, ...]
    """
    lookup = defaultdict(list)
    
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            word = rec.get("word", "").lower()
            for cand in rec.get("top_k_atoms", []):
                lookup[word].append({
                    "atom": cand["atom"],
                    "cos_raw": cand["cos_raw"],
                    "rank": cand["rank"],
                    "assigned_atom": rec.get("assigned_atom", ""),
                })
    
    return dict(lookup)


def load_dictionary(path: str) -> Dict[str, Dict]:
    with open(path) as f:
        raw = json.load(f)
    return raw.get("concepts", raw)


# ============================================================
# V4 Re-Grounding
# ============================================================

def reground_v4(verb_lemma: str,
                v4_lookup: Dict[str, List[Dict]],
                min_score: float = 0.45) -> List[Dict]:
    """
    Simulate grounding using v4 candidates.
    
    Look up verb_lemma in v4 candidate lookup.
    Apply POS Guard (block NAT/MAT/PRP/SPA atoms).
    Return sorted candidates.
    """
    candidates = v4_lookup.get(verb_lemma.lower(), [])
    
    if not candidates:
        return []
    
    # POS Guard
    filtered = []
    for c in candidates:
        cat = c["atom"].split(".")[0] if "." in c["atom"] else ""
        if cat in POS_GUARD_BLOCKED:
            continue
        filtered.append(c)
    
    # Sort by cos_raw descending
    filtered.sort(key=lambda x: -x["cos_raw"])
    
    return filtered


def compare_grounding(v3_edges: List[Dict],
                      v4_lookup: Dict[str, List[Dict]],
                      dictionary: Dict) -> Dict[str, Any]:
    """
    For each v3 edge, simulate v4 grounding and compare.
    """
    results = []
    
    # Counters
    v3_grounded = 0
    v3_ungrounded = 0
    v3_lightverb = 0
    v4_grounded = 0
    v4_ungrounded = 0
    v4_lightverb = 0
    
    # Category tracking
    v3_cat_count = Counter()
    v4_cat_count = Counter()
    
    # Agreement tracking
    agree = 0
    disagree = 0
    v4_new_ground = 0  # was ungrounded in v3, grounded in v4
    v4_lost_ground = 0  # was grounded in v3, ungrounded in v4
    
    # Category mismatch tracking
    v3_cat_mismatch = 0
    v4_cat_mismatch = 0
    
    for edge in v3_edges:
        verb = edge.get("verb_lemma", "")
        v3_status = edge.get("grounding_status", "UNGROUNDED")
        v3_atom = edge.get("atom", "UNGROUNDED")
        v3_candidates = edge.get("atom_candidates", [])
        
        # V3 classification
        is_light = verb.lower() in LIGHT_VERB_STOPLIST
        
        if is_light:
            v3_lightverb += 1
            v4_lightverb += 1
            results.append({
                "verb": verb,
                "v3_status": "LIGHTVERB",
                "v4_status": "LIGHTVERB",
                "v3_atom": v3_atom,
                "v4_atom": "LIGHTVERB",
                "match": True,
            })
            continue
        
        if v3_status == "GROUNDED":
            v3_grounded += 1
            v3_cat = v3_atom.split(".")[0] if "." in v3_atom else ""
            v3_cat_count[v3_cat] += 1
        else:
            v3_ungrounded += 1
        
        # V4 re-grounding
        v4_cands = reground_v4(verb, v4_lookup)
        
        if v4_cands:
            v4_top = v4_cands[0]["atom"]
            v4_status = "GROUNDED"
            v4_grounded += 1
            v4_cat = v4_top.split(".")[0] if "." in v4_top else ""
            v4_cat_count[v4_cat] += 1
        else:
            v4_top = "UNGROUNDED"
            v4_status = "UNGROUNDED"
            v4_ungrounded += 1
            v4_cat = ""
        
        # Compare
        match = (v3_atom == v4_top)
        
        if v3_status == "GROUNDED" and v4_status == "GROUNDED":
            if match:
                agree += 1
            else:
                disagree += 1
        elif v3_status != "GROUNDED" and v4_status == "GROUNDED":
            v4_new_ground += 1
        elif v3_status == "GROUNDED" and v4_status != "GROUNDED":
            v4_lost_ground += 1
        
        # Category mismatch check (verb grounded to noun-category atom)
        if v3_status == "GROUNDED" and v3_cat in POS_GUARD_BLOCKED:
            v3_cat_mismatch += 1
        if v4_status == "GROUNDED" and v4_cat in POS_GUARD_BLOCKED:
            v4_cat_mismatch += 1
        
        results.append({
            "verb": verb,
            "v3_status": v3_status,
            "v4_status": v4_status,
            "v3_atom": v3_atom,
            "v4_atom": v4_top,
            "v4_top3": [c["atom"] for c in v4_cands[:3]],
            "v4_cos": v4_cands[0]["cos_raw"] if v4_cands else 0,
            "match": match,
            "sentence": edge.get("text_ref", "")[:100],
        })
    
    total = len(v3_edges)
    total_non_light = total - v3_lightverb
    
    # Coverage per atom category
    all_cats = sorted(set(list(v3_cat_count.keys()) + list(v4_cat_count.keys())))
    category_comparison = []
    for cat in all_cats:
        if not cat:
            continue
        category_comparison.append({
            "category": cat,
            "v3_count": v3_cat_count.get(cat, 0),
            "v4_count": v4_cat_count.get(cat, 0),
            "diff": v4_cat_count.get(cat, 0) - v3_cat_count.get(cat, 0),
        })
    
    return {
        "total_edges": total,
        "total_non_light": total_non_light,
        "v3": {
            "grounded": v3_grounded,
            "ungrounded": v3_ungrounded,
            "lightverb": v3_lightverb,
            "grounding_rate": round(v3_grounded / total_non_light * 100, 1) if total_non_light > 0 else 0,
            "cat_mismatch": v3_cat_mismatch,
        },
        "v4": {
            "grounded": v4_grounded,
            "ungrounded": v4_ungrounded,
            "lightverb": v4_lightverb,
            "grounding_rate": round(v4_grounded / total_non_light * 100, 1) if total_non_light > 0 else 0,
            "cat_mismatch": v4_cat_mismatch,
        },
        "comparison": {
            "agree": agree,
            "disagree": disagree,
            "v4_new_ground": v4_new_ground,
            "v4_lost_ground": v4_lost_ground,
            "agreement_rate": round(agree / (agree + disagree) * 100, 1) if (agree + disagree) > 0 else 0,
        },
        "category_comparison": category_comparison,
        "details": results,
    }


# ============================================================
# Report Generation
# ============================================================

def write_report(comparison: Dict, out_dir: str):
    """Write obsC_comparison_v3_vs_v4.md"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    c = comparison
    v3 = c["v3"]
    v4 = c["v4"]
    comp = c["comparison"]
    
    lines = []
    lines.append("# ESDE Synapse v4 — Task 3: ObsC Grounding Comparison (v3 vs v4)")
    lines.append("")
    lines.append(f"**Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**Phase**: Analysis-only (no permanent changes)")
    lines.append(f"**Method**: Re-ground existing v3 edges using A1-derived v4 candidates")
    lines.append("")
    
    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Synapse v3 | v4 Candidates | Delta |")
    lines.append(f"|--------|-----------|---------------|-------|")
    lines.append(f"| Grounded | {v3['grounded']} | {v4['grounded']} | "
                 f"{v4['grounded'] - v3['grounded']:+d} |")
    lines.append(f"| Ungrounded | {v3['ungrounded']} | {v4['ungrounded']} | "
                 f"{v4['ungrounded'] - v3['ungrounded']:+d} |")
    lines.append(f"| Light verb (skipped) | {v3['lightverb']} | {v4['lightverb']} | 0 |")
    lines.append(f"| **Grounding rate** | **{v3['grounding_rate']}%** | "
                 f"**{v4['grounding_rate']}%** | "
                 f"**{v4['grounding_rate'] - v3['grounding_rate']:+.1f}pp** |")
    lines.append(f"| Category mismatch | {v3['cat_mismatch']} | {v4['cat_mismatch']} | "
                 f"{v4['cat_mismatch'] - v3['cat_mismatch']:+d} |")
    lines.append("")
    
    # Agreement
    lines.append("## Agreement Analysis")
    lines.append("")
    lines.append(f"| Metric | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Both grounded, same atom | {comp['agree']} |")
    lines.append(f"| Both grounded, different atom | {comp['disagree']} |")
    lines.append(f"| v4 newly grounded (v3 was ungrounded) | {comp['v4_new_ground']} |")
    lines.append(f"| v4 lost grounding (v3 was grounded) | {comp['v4_lost_ground']} |")
    lines.append(f"| **Agreement rate** (when both grounded) | **{comp['agreement_rate']}%** |")
    lines.append("")
    
    # Category coverage
    lines.append("## Category Coverage Comparison")
    lines.append("")
    lines.append("| Category | v3 | v4 | Diff |")
    lines.append("|----------|----|----|------|")
    for cc in sorted(c["category_comparison"], key=lambda x: -abs(x["diff"])):
        lines.append(f"| {cc['category']} | {cc['v3_count']} | {cc['v4_count']} | "
                     f"{cc['diff']:+d} |")
    lines.append("")
    
    # Decision rule
    lines.append("## Decision Rule Evaluation")
    lines.append("")
    
    gr_improved = v4["grounding_rate"] > v3["grounding_rate"]
    
    # Check ACT/CHG improvement
    act_chg_improved = False
    for cc in c["category_comparison"]:
        if cc["category"] in ("ACT", "CHG") and cc["diff"] > 0:
            act_chg_improved = True
    
    if gr_improved:
        lines.append(f"✅ Grounding rate improved: {v3['grounding_rate']}% → {v4['grounding_rate']}%")
    else:
        lines.append(f"❌ Grounding rate did not improve: {v3['grounding_rate']}% → {v4['grounding_rate']}%")
    
    if act_chg_improved:
        lines.append(f"✅ High-impact verb categories (ACT, CHG) improved")
    else:
        lines.append(f"⚠ ACT/CHG categories did not materially improve")
    
    lines.append("")
    if gr_improved or act_chg_improved:
        lines.append("**Verdict**: v4 candidate approach shows viability for Synapse improvement.")
    else:
        lines.append("**Verdict**: v4 candidate approach does not yet show clear improvement. "
                     "Further investigation needed.")
    lines.append("")
    
    # Sample disagreements
    disagree_samples = [d for d in c["details"]
                       if d["v3_status"] == "GROUNDED" and d["v4_status"] == "GROUNDED"
                       and not d["match"]][:30]
    
    if disagree_samples:
        lines.append("## Sample Disagreements (v3 ≠ v4, both grounded)")
        lines.append("")
        lines.append("| Verb | v3 Atom | v4 Atom | v4 cos | Sentence |")
        lines.append("|------|---------|---------|--------|----------|")
        for d in disagree_samples:
            sent = d["sentence"][:60].replace("|", "\\|")
            lines.append(f"| {d['verb']} | {d['v3_atom']} | {d['v4_atom']} | "
                        f"{d['v4_cos']:.4f} | {sent}... |")
        lines.append("")
    
    # v4 newly grounded samples
    new_ground = [d for d in c["details"]
                  if d["v3_status"] != "GROUNDED" and d["v4_status"] == "GROUNDED"][:20]
    
    if new_ground:
        lines.append("## Sample v4 New Groundings (v3 was ungrounded)")
        lines.append("")
        lines.append("| Verb | v4 Atom | v4 cos | Sentence |")
        lines.append("|------|---------|--------|----------|")
        for d in new_ground:
            sent = d["sentence"][:60].replace("|", "\\|")
            lines.append(f"| {d['verb']} | {d['v4_atom']} | "
                        f"{d['v4_cos']:.4f} | {sent}... |")
        lines.append("")
    
    report_path = out / "obsC_comparison_v3_vs_v4.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✅ Report: {report_path}")


def write_category_csv(comparison: Dict, out_dir: str):
    """Write obsC_category_diff.csv"""
    import csv
    out = Path(out_dir)
    csv_path = out / "obsC_category_diff.csv"
    
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category", "v3_count", "v4_count", "diff"])
        w.writeheader()
        for cc in sorted(comparison["category_comparison"], key=lambda x: x["category"]):
            w.writerow(cc)
    
    print(f"  ✅ CSV: {csv_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="ESDE Synapse v4 — Task 3: ObsC Grounding Comparison"
    )
    parser.add_argument("--v3-edges-dir", required=True,
                        help="Directory containing v3 *_edges.jsonl files")
    parser.add_argument("--v4-candidates", required=True,
                        help="synapse_v4_candidates.jsonl from Task 2")
    parser.add_argument("--dictionary", required=True,
                        help="esde_dictionary.json")
    parser.add_argument("--out-dir", default="synapse_v4_report",
                        help="Output directory")
    args = parser.parse_args()
    
    print("=" * 70)
    print("  ESDE Synapse v4 — Task 3: ObsC Grounding Comparison")
    print("  Analysis-only — no permanent changes")
    print("=" * 70)
    
    # Load
    print("\n[1/3] Loading v3 edges...")
    v3_edges = load_v3_edges(args.v3_edges_dir)
    print(f"  Total edges: {len(v3_edges)}")
    
    grounded = sum(1 for e in v3_edges if e.get("grounding_status") == "GROUNDED")
    ungrounded = sum(1 for e in v3_edges if e.get("grounding_status") == "UNGROUNDED")
    light = sum(1 for e in v3_edges if e.get("grounding_status") == "UNGROUNDED_LIGHTVERB")
    print(f"  v3: grounded={grounded}, ungrounded={ungrounded}, lightverb={light}")
    
    print("\n[2/3] Loading v4 candidates...")
    v4_lookup = load_v4_candidates(args.v4_candidates)
    print(f"  Unique words in v4: {len(v4_lookup)}")
    
    print("\n[3/3] Loading dictionary...")
    dictionary = load_dictionary(args.dictionary)
    print(f"  Atoms: {len(dictionary)}")
    
    # Compare
    print("\n" + "=" * 70)
    print("  Comparing v3 vs v4 grounding...")
    print("=" * 70)
    
    comparison = compare_grounding(v3_edges, v4_lookup, dictionary)
    
    v3s = comparison["v3"]
    v4s = comparison["v4"]
    comp = comparison["comparison"]
    
    print(f"\n  {'Metric':<30s} {'v3':>8s} {'v4':>8s} {'Delta':>8s}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'Grounding rate':<30s} {v3s['grounding_rate']:>7.1f}% "
          f"{v4s['grounding_rate']:>7.1f}% {v4s['grounding_rate']-v3s['grounding_rate']:>+7.1f}pp")
    print(f"  {'Category mismatch':<30s} {v3s['cat_mismatch']:>8d} "
          f"{v4s['cat_mismatch']:>8d} {v4s['cat_mismatch']-v3s['cat_mismatch']:>+8d}")
    print(f"  {'Agreement (both grounded)':<30s} {'':>8s} "
          f"{comp['agreement_rate']:>7.1f}% {'':>8s}")
    print(f"  {'v4 new groundings':<30s} {'':>8s} {comp['v4_new_ground']:>8d}")
    print(f"  {'v4 lost groundings':<30s} {'':>8s} {comp['v4_lost_ground']:>8d}")
    
    # Write outputs
    print("\n" + "=" * 70)
    print("  Generating Reports")
    print("=" * 70)
    
    write_report(comparison, args.out_dir)
    write_category_csv(comparison, args.out_dir)
    
    print("\n" + "=" * 70)
    print("  COMPLETE — Task 3")
    print("=" * 70)


if __name__ == "__main__":
    main()
