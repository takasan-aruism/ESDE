#!/usr/bin/env python3
"""
ESDE Lexicon v2 — Core Pool Cross-Statistics
=============================================
Same 10-column report as wn_cross_stats.py, but computed ONLY on Core Pool words.
This measures "the world the Mapper actually sees" — not the full expansion.

Comparison with full-expansion stats reveals how much Core/Deviation split
improves orthogonality.

Usage:
  python3 wn_core_stats.py --dir lexicon/ --dictionary esde_dictionary.json --out core_report.csv
"""

import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict, Counter


def load_lexicon_entries(lexicon_dir: str, skip_merged: bool = True) -> dict:
    """Load all lexicon entry JSONs. Optionally skip merged atoms."""
    entries = {}
    skipped = 0
    for f in sorted(Path(lexicon_dir).glob("*.json")):
        if f.name.startswith("_"):
            continue
        with open(f) as fh:
            data = json.load(fh)
        if "atom" in data:
            if skip_merged and data.get("status") == "merged":
                skipped += 1
                continue
            entries[data["atom"]] = data
    if skipped:
        print(f"  Skipped {skipped} merged atom(s)")
    return entries


def load_dictionary(dict_path: str) -> dict:
    with open(dict_path) as f:
        raw = json.load(f)
    if "concepts" in raw and isinstance(raw["concepts"], dict):
        return raw["concepts"]
    return raw


def compute_core_stats(entries: dict, dictionary: dict):
    """Compute cross-atom stats using ONLY core_pool words."""
    
    # Build reverse index: (lemma_lower, pos) → set of atom_ids (core only)
    core_index = defaultdict(set)
    for atom_id, entry in entries.items():
        for w in entry.get("core_pool", {}).get("words", []):
            key = (w["w"].lower(), w["pos"])
            core_index[key].add(atom_id)
    
    total_atoms = len(entries)
    generic_thresholds = {
        "5pct": max(1, int(total_atoms * 0.05)),
        "10pct": max(1, int(total_atoms * 0.10)),
        "20pct": max(1, int(total_atoms * 0.20)),
    }
    
    rows = []
    
    for atom_id, entry in sorted(entries.items()):
        core_words = entry.get("core_pool", {}).get("words", [])
        dev_words = entry.get("deviation_pool", {}).get("words", [])
        dict_entry = dictionary.get(atom_id, {})
        category = entry.get("category", dict_entry.get("category", "?"))
        
        core_count = len(core_words)
        dev_count = len(dev_words)
        total = core_count + dev_count
        
        # POS distribution (core only)
        pos_counts = defaultdict(int)
        for w in core_words:
            pos_counts[w["pos"]] += 1
        total_pos = sum(pos_counts.values()) or 1
        
        # Unique keys (core only — not in any other atom's core)
        my_core_keys = set((w["w"].lower(), w["pos"]) for w in core_words)
        unique_count = sum(1 for k in my_core_keys if len(core_index.get(k, set())) == 1)
        unique_ratio = round(unique_count / core_count * 100, 1) if core_count > 0 else 0
        
        # Atoms per word (core only)
        apw_list = [len(core_index.get((w["w"].lower(), w["pos"]), set())) for w in core_words]
        mean_apw = round(sum(apw_list) / len(apw_list), 2) if apw_list else 0
        if apw_list:
            sorted_apw = sorted(apw_list)
            p95_apw = sorted_apw[min(int(len(sorted_apw) * 0.95), len(sorted_apw) - 1)]
        else:
            p95_apw = 0
        
        # Generic counts (core only)
        generic_counts = {}
        for label, threshold in generic_thresholds.items():
            count = sum(1 for w in core_words
                       if len(core_index.get((w["w"].lower(), w["pos"]), set())) >= threshold)
            generic_counts[label] = count
        
        # Top5 overlap (core only)
        overlap_counter = Counter()
        for key in my_core_keys:
            for other_atom in core_index.get(key, set()):
                if other_atom != atom_id:
                    overlap_counter[other_atom] += 1
        
        top5 = []
        for other_atom, ovl_count in overlap_counter.most_common(5):
            other_entry = entries.get(other_atom, {})
            other_keys = set(
                (w["w"].lower(), w["pos"])
                for w in other_entry.get("core_pool", {}).get("words", [])
            )
            union_size = len(my_core_keys | other_keys) or 1
            jaccard = round(ovl_count / union_size, 3)
            top5.append({"atom": other_atom, "overlap": ovl_count, "jaccard": jaccard})
        
        # Symmetry leak (core only — use dictionary for rewired sym_pairs)
        sym_pair = dict_entry.get("symmetric_pair", entry.get("symmetric_pair", ""))
        sym_leak = {"pair": sym_pair, "overlap_keys": 0}
        if sym_pair and sym_pair in entries:
            pair_keys = set(
                (w["w"].lower(), w["pos"])
                for w in entries[sym_pair].get("core_pool", {}).get("words", [])
            )
            sym_leak["overlap_keys"] = len(my_core_keys & pair_keys)
        
        rows.append({
            "atom_id": atom_id,
            "category": category,
            "core_count": core_count,
            "dev_count": dev_count,
            "core_ratio_pct": round(core_count / total * 100, 1) if total > 0 else 0,
            "zero_core": core_count == 0,
            "thin_core": 0 < core_count < 15,
            "pos_n_pct": round(pos_counts.get("n", 0) / total_pos * 100, 1),
            "pos_v_pct": round(pos_counts.get("v", 0) / total_pos * 100, 1),
            "pos_adj_pct": round(pos_counts.get("adj", 0) / total_pos * 100, 1),
            "unique_keys": unique_count,
            "unique_ratio_pct": unique_ratio,
            "mean_atoms_per_word": mean_apw,
            "p95_atoms_per_word": p95_apw,
            "generic_at_5pct": generic_counts["5pct"],
            "generic_at_10pct": generic_counts["10pct"],
            "generic_at_20pct": generic_counts["20pct"],
            "top1_overlap_atom": top5[0]["atom"] if top5 else "",
            "top1_overlap_count": top5[0]["overlap"] if top5 else 0,
            "top1_jaccard": top5[0]["jaccard"] if top5 else 0,
            "sym_pair": sym_leak["pair"],
            "sym_core_overlap": sym_leak["overlap_keys"],
        })
    
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Lexicon entries directory")
    parser.add_argument("--dictionary", required=True, help="esde_dictionary.json (or v2)")
    parser.add_argument("--out", default="core_report.csv", help="Output CSV")
    parser.add_argument("--include-merged", action="store_true",
                        help="Include merged atoms (default: skip them)")
    args = parser.parse_args()
    
    entries = load_lexicon_entries(args.dir, skip_merged=not args.include_merged)
    dictionary = load_dictionary(args.dictionary)
    
    print(f"Loaded {len(entries)} lexicon entries")
    
    rows = compute_core_stats(entries, dictionary)
    
    if rows:
        with open(args.out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    
    # Compare before/after
    print(f"\n{'='*70}")
    print(f"  CORE POOL STATISTICS (Mapper's world)")
    print(f"{'='*70}")
    
    core_counts = [r["core_count"] for r in rows]
    dev_counts = [r["dev_count"] for r in rows]
    zeros = sum(1 for r in rows if r["zero_core"])
    thins = sum(1 for r in rows if r["thin_core"])
    
    print(f"\n  ■ Supply (Core only):")
    print(f"    Atoms: {len(rows)}")
    print(f"    Avg core/atom: {sum(core_counts)/len(core_counts):.1f}")
    print(f"    Avg dev/atom: {sum(dev_counts)/len(dev_counts):.1f}")
    print(f"    Zero core: {zeros}")
    print(f"    Thin core (<15): {thins}")
    
    print(f"\n  ■ Orthogonality (Core only):")
    apw = [r["mean_atoms_per_word"] for r in rows if r["core_count"] > 0]
    high_apw = sum(1 for a in apw if a > 8)
    print(f"    mean_APW median: {sorted(apw)[len(apw)//2]:.1f}")
    print(f"    mean_APW > 8 (contaminated): {high_apw}")
    
    low_uniq = sum(1 for r in rows if r["unique_ratio_pct"] < 5 and r["core_count"] > 0)
    print(f"    unique_ratio < 5%: {low_uniq}")
    
    print(f"\n  ■ Coordinate Collapse (Core only, Jaccard > 0.4):")
    collapse = [(r["atom_id"], r["top1_overlap_atom"], r["top1_jaccard"])
                for r in rows if r["top1_jaccard"] > 0.4]
    for a, b, j in sorted(collapse, key=lambda x: -x[2])[:15]:
        print(f"    {a:30s} ↔ {b:30s}  J={j:.3f}")
    
    print(f"\n  ■ Symmetry Leak (Core only):")
    leaks = [(r["atom_id"], r["sym_pair"], r["sym_core_overlap"])
             for r in rows if r["sym_core_overlap"] > 10]
    for a, b, ovl in sorted(leaks, key=lambda x: -x[2])[:10]:
        print(f"    {a:30s} ↔ {b:25s}  core_shared={ovl}")
    
    if not leaks:
        print(f"    No significant core-level symmetry leaks (all ≤10)")
    
    print(f"\n  ✅ Report: {args.out}")


if __name__ == "__main__":
    main()
