#!/usr/bin/env python3
"""
ESDE Lexicon v2 — Cross-Atom Statistics Report
================================================
Reads batch expand results → computes GPT 10-column report.

Columns (GPT/Gemini consolidated):
  1. Atom ID / Category
  2. SeedCount / SeedLemmaCount
  3. TotalKeys (unique lemma::pos)
  4. POS% (n/v/adj/adv)
  5. SourceRatio% (seed/hypernym/hyponym/deriv/sibling/antonym...)
  6. UniqueKeysCount (lemma::pos not in any other atom)
  7. MeanAtomsPerWord / P95AtomsPerWord
  8. GenericCount@5% / @10% / @20%
  9. Top5OverlapAtoms (+Jaccard)
  10. SymmetryLeak (PairOverlapKeys / fromAntonym / fromSibling)

Plus "what-if" columns:
  - Coverage if Generic@K% removed

Usage:
  python3 wn_cross_stats.py --dir expanded/ --dictionary esde_dictionary.json --out report.csv
"""

import json
import csv
import sys
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import math


def load_expanded(expand_dir: str) -> dict:
    """Load all individual atom expand results."""
    results = {}
    expand_path = Path(expand_dir)
    
    for f in sorted(expand_path.glob("*.json")):
        if f.name.startswith("_"):
            continue
        with open(f) as fh:
            data = json.load(fh)
        if "atom_id" in data:
            results[data["atom_id"]] = data
    
    return results


def load_dictionary(dict_path: str) -> dict:
    """Load esde_dictionary.json for symmetric pair info."""
    with open(dict_path) as f:
        raw = json.load(f)
    if "concepts" in raw and isinstance(raw["concepts"], dict):
        return raw["concepts"]
    return raw


def build_word_to_atoms_index(results: dict) -> dict:
    """Build reverse index: (lemma_lower, pos) → set of atom_ids."""
    index = defaultdict(set)
    for atom_id, data in results.items():
        for c in data.get("candidates", []):
            key = (c["lemma"].lower(), c["pos"])
            index[key].add(atom_id)
    return dict(index)


def compute_stats(results: dict, dictionary: dict):
    """Compute all 10 columns + what-if analysis."""
    
    word_to_atoms = build_word_to_atoms_index(results)
    total_atoms = len(results)
    
    # Thresholds for generic word detection
    generic_thresholds = {
        "5pct": max(1, int(total_atoms * 0.05)),
        "10pct": max(1, int(total_atoms * 0.10)),
        "20pct": max(1, int(total_atoms * 0.20)),
    }
    
    rows = []
    
    for atom_id, data in sorted(results.items()):
        candidates = data.get("candidates", [])
        category = data.get("category", "?")
        seed_count = data.get("seed_count", 0)
        total_keys = data.get("total_keys", len(candidates))
        
        # ── Col 2: SeedLemmaCount ──
        seed_lemma_count = sum(
            1 for s in data.get("steps", [])
            if s["step"] == "0_seed"
            for _ in range(s["added"])
        )
        # Actually just use the step added count
        for s in data.get("steps", []):
            if s["step"] == "0_seed":
                seed_lemma_count = s["added"]
                break
        
        # ── Col 4: POS% ──
        pos_counts = data.get("pos_counts", {})
        total_pos = sum(pos_counts.values()) or 1
        pos_pct = {p: round(c / total_pos * 100, 1) for p, c in pos_counts.items()}
        
        # ── Col 5: SourceRatio% ──
        source_counts = data.get("source_counts", {})
        total_source = sum(source_counts.values()) or 1
        source_pct = {}
        for src, cnt in source_counts.items():
            # Simplify step names
            simple = src.split("_", 1)[1] if "_" in src else src
            source_pct[simple] = round(cnt / total_source * 100, 1)
        
        # Sibling ratio specifically
        sibling_count = source_counts.get("10_sibling", 0)
        sibling_ratio = round(sibling_count / total_source * 100, 1) if total_source > 0 else 0
        
        # ── Col 6: UniqueKeysCount ──
        unique_count = 0
        for c in candidates:
            key = (c["lemma"].lower(), c["pos"])
            if len(word_to_atoms.get(key, set())) == 1:
                unique_count += 1
        unique_ratio = round(unique_count / total_keys * 100, 1) if total_keys > 0 else 0
        
        # ── Col 7: MeanAtomsPerWord / P95AtomsPerWord ──
        atoms_per_word = []
        for c in candidates:
            key = (c["lemma"].lower(), c["pos"])
            atoms_per_word.append(len(word_to_atoms.get(key, set())))
        
        mean_apw = round(sum(atoms_per_word) / len(atoms_per_word), 2) if atoms_per_word else 0
        if atoms_per_word:
            sorted_apw = sorted(atoms_per_word)
            p95_idx = int(len(sorted_apw) * 0.95)
            p95_apw = sorted_apw[min(p95_idx, len(sorted_apw) - 1)]
        else:
            p95_apw = 0
        
        # ── Col 8: GenericCount@K% ──
        generic_counts = {}
        for label, threshold in generic_thresholds.items():
            count = sum(
                1 for c in candidates
                if len(word_to_atoms.get((c["lemma"].lower(), c["pos"]), set())) >= threshold
            )
            generic_counts[label] = count
        
        # ── Col 9: Top5OverlapAtoms + Jaccard ──
        # Count overlapping words per other atom
        overlap_counter = Counter()
        my_keys = set((c["lemma"].lower(), c["pos"]) for c in candidates)
        for key in my_keys:
            for other_atom in word_to_atoms.get(key, set()):
                if other_atom != atom_id:
                    overlap_counter[other_atom] += 1
        
        top5_overlap = []
        for other_atom, overlap_count in overlap_counter.most_common(5):
            other_data = results.get(other_atom, {})
            other_keys = set(
                (c["lemma"].lower(), c["pos"])
                for c in other_data.get("candidates", [])
            )
            union_size = len(my_keys | other_keys) or 1
            jaccard = round(overlap_count / union_size, 3)
            top5_overlap.append({
                "atom": other_atom,
                "overlap": overlap_count,
                "jaccard": jaccard,
            })
        
        # ── Col 10: SymmetryLeak ──
        sym_pair = dictionary.get(atom_id, {}).get("symmetric_pair", "")
        sym_leak = {"pair": sym_pair, "overlap_keys": 0,
                    "from_antonym": 0, "from_sibling": 0}
        
        if sym_pair and sym_pair in results:
            pair_data = results[sym_pair]
            pair_keys = set(
                (c["lemma"].lower(), c["pos"])
                for c in pair_data.get("candidates", [])
            )
            overlap_keys = my_keys & pair_keys
            sym_leak["overlap_keys"] = len(overlap_keys)
            
            # Classify overlap sources
            for c in candidates:
                key = (c["lemma"].lower(), c["pos"])
                if key in overlap_keys:
                    for path in c.get("paths", []):
                        if "antonym" in path.get("relation", ""):
                            sym_leak["from_antonym"] += 1
                        if "sibling" in path.get("relation", ""):
                            sym_leak["from_sibling"] += 1
        
        # ── Thinness flags ──
        thin_flag = total_keys < 15
        zero_flag = total_keys == 0
        
        # ── What-if: coverage after removing generics ──
        whatif = {}
        for label, threshold in generic_thresholds.items():
            remaining = total_keys - generic_counts[label]
            whatif[f"remaining_after_{label}"] = remaining
        
        rows.append({
            "atom_id": atom_id,
            "category": category,
            "seed_count": seed_count,
            "seed_lemma_count": seed_lemma_count,
            "total_keys": total_keys,
            "zero_flag": zero_flag,
            "thin_flag": thin_flag,
            "pos_n_pct": pos_pct.get("n", 0),
            "pos_v_pct": pos_pct.get("v", 0),
            "pos_adj_pct": pos_pct.get("adj", 0),
            "pos_adv_pct": pos_pct.get("adv", 0),
            "sibling_ratio_pct": sibling_ratio,
            "unique_keys": unique_count,
            "unique_ratio_pct": unique_ratio,
            "mean_atoms_per_word": mean_apw,
            "p95_atoms_per_word": p95_apw,
            "generic_at_5pct": generic_counts["5pct"],
            "generic_at_10pct": generic_counts["10pct"],
            "generic_at_20pct": generic_counts["20pct"],
            "top1_overlap_atom": top5_overlap[0]["atom"] if top5_overlap else "",
            "top1_overlap_count": top5_overlap[0]["overlap"] if top5_overlap else 0,
            "top1_jaccard": top5_overlap[0]["jaccard"] if top5_overlap else 0,
            "top5_overlap_json": json.dumps(top5_overlap),
            "sym_pair": sym_leak["pair"],
            "sym_overlap_keys": sym_leak["overlap_keys"],
            "sym_from_antonym": sym_leak["from_antonym"],
            "sym_from_sibling": sym_leak["from_sibling"],
            "remaining_after_5pct": whatif.get("remaining_after_5pct", 0),
            "remaining_after_10pct": whatif.get("remaining_after_10pct", 0),
            "remaining_after_20pct": whatif.get("remaining_after_20pct", 0),
        })
    
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory with expanded atom JSONs")
    parser.add_argument("--dictionary", required=True, help="esde_dictionary.json path")
    parser.add_argument("--out", default="report.csv", help="Output CSV path")
    args = parser.parse_args()
    
    print("Loading expanded results...")
    results = load_expanded(args.dir)
    print(f"  Loaded {len(results)} atoms")
    
    print("Loading dictionary...")
    dictionary = load_dictionary(args.dictionary)
    print(f"  Loaded {len(dictionary)} atom definitions")
    
    print("Computing cross-atom statistics...")
    rows = compute_stats(results, dictionary)
    
    # Write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(args.out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"  CROSS-ATOM STATISTICS REPORT")
    print(f"{'='*70}")
    
    total_keys = [r["total_keys"] for r in rows]
    zeros = sum(1 for r in rows if r["zero_flag"])
    thins = sum(1 for r in rows if r["thin_flag"] and not r["zero_flag"])
    
    print(f"\n  Atoms: {len(rows)}")
    print(f"  Total candidates across all atoms: {sum(total_keys)}")
    print(f"  Avg candidates/atom: {sum(total_keys)/len(total_keys):.1f}")
    print(f"  Median: {sorted(total_keys)[len(total_keys)//2]}")
    print(f"  Zero atoms: {zeros}")
    print(f"  Thin atoms (1-14): {thins}")
    
    # Sibling contamination
    high_sibling = [r for r in rows if r["sibling_ratio_pct"] > 40]
    print(f"\n  High sibling ratio (>40%): {len(high_sibling)} atoms")
    for r in sorted(high_sibling, key=lambda x: -x["sibling_ratio_pct"])[:10]:
        print(f"    {r['atom_id']:25s}  sibling={r['sibling_ratio_pct']}%  total={r['total_keys']}")
    
    # Symmetry leaks
    leaks = [r for r in rows if r["sym_overlap_keys"] > 5]
    print(f"\n  Symmetry leaks (>5 shared keys): {len(leaks)} pairs")
    for r in sorted(leaks, key=lambda x: -x["sym_overlap_keys"])[:10]:
        print(f"    {r['atom_id']:25s} ↔ {r['sym_pair']:25s}  shared={r['sym_overlap_keys']}  ant={r['sym_from_antonym']}  sib={r['sym_from_sibling']}")
    
    # Generic word distribution
    g5 = sum(r["generic_at_5pct"] for r in rows)
    g10 = sum(r["generic_at_10pct"] for r in rows)
    g20 = sum(r["generic_at_20pct"] for r in rows)
    print(f"\n  Generic words (appear in ≥K% of atoms):")
    print(f"    @5%:  {g5} total occurrences")
    print(f"    @10%: {g10} total occurrences")
    print(f"    @20%: {g20} total occurrences")
    
    # What-if coverage
    avg_remaining_5 = sum(r["remaining_after_5pct"] for r in rows) / len(rows)
    avg_remaining_10 = sum(r["remaining_after_10pct"] for r in rows) / len(rows)
    avg_remaining_20 = sum(r["remaining_after_20pct"] for r in rows) / len(rows)
    avg_total = sum(total_keys) / len(total_keys)
    print(f"\n  What-if: avg candidates remaining after removing generics:")
    print(f"    Remove @5%:  {avg_remaining_5:.1f} / {avg_total:.1f} ({avg_remaining_5/avg_total*100:.0f}%)")
    print(f"    Remove @10%: {avg_remaining_10:.1f} / {avg_total:.1f} ({avg_remaining_10/avg_total*100:.0f}%)")
    print(f"    Remove @20%: {avg_remaining_20:.1f} / {avg_total:.1f} ({avg_remaining_20/avg_total*100:.0f}%)")
    
    # Category breakdown
    print(f"\n  By category:")
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    for cat, cat_rows in sorted(by_cat.items()):
        cat_totals = [r["total_keys"] for r in cat_rows]
        cat_zeros = sum(1 for t in cat_totals if t == 0)
        cat_uniq = sum(r["unique_ratio_pct"] for r in cat_rows) / len(cat_rows)
        print(f"    {cat:6s}: {len(cat_rows):3d} atoms  avg={sum(cat_totals)/len(cat_totals):5.1f}  zeros={cat_zeros}  avg_unique={cat_uniq:.0f}%")
    
    print(f"\n  ✅ Report saved: {args.out}")


if __name__ == "__main__":
    main()