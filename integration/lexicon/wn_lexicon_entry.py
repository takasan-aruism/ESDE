#!/usr/bin/env python3
"""
ESDE Lexicon v2 — Lexicon Entry Generator
==========================================
Reads batch-expanded atom results → produces Core/Deviation split entries.

Architecture: "二層＋三状態" (3AI consolidated design)
  - Core Pool: seed, hyponym_d1, derivational, similar_to, antonym
    → Mapper input (座標決定用)
  - Deviation Pool: sibling, hypernym, hyponym_d2+, pertainym, verb_group
    → Observation data (偏りDB, Phase7/9 fuel)
  - Status: proposed → audited → core

Usage:
  # From pre-expanded individual atom JSONs
  python3 wn_lexicon_entry.py --dir expanded/ --dictionary esde_dictionary.json --outdir lexicon/

  # Single atom test
  python3 wn_lexicon_entry.py --dir expanded/ --dictionary esde_dictionary.json --outdir lexicon/ --atoms EMO.like
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

# ============================================================
# Core/Deviation Step Classification (3AI agreed)
# ============================================================

CORE_STEPS = {
    "0_seed",           # Definition nucleus
    "3_hyponym_d1",     # Direct concretization
    "6_derivational",   # Cross-POS same concept
    "7_similar_to",     # Synonym neighborhood (effective for adj)
    "9_antonym",        # Symmetry pair's destructive side
}

DEVIATION_STEPS = {
    "2_hypernym_d1",    # Too generic (上位概念)
    "4_hyponym_d2",     # Too deep concretization
    "5_hyponym_d3",     # Even deeper
    "8_also_see",       # Loose association
    "10_sibling",       # MAIN contamination source & MAIN information source
    "11_pertainym",     # Sporadic
    "12_verb_group",    # Sporadic
}


def classify_candidate(candidate: dict) -> str:
    """
    Classify a candidate word as 'core' or 'deviation' based on its paths.
    
    Rule: If ANY path is from a Core step, the word is Core.
    (A word found via both seed and sibling → Core, because seed is sufficient evidence)
    """
    for path in candidate.get("paths", []):
        step = path.get("step", "")
        if step in CORE_STEPS:
            return "core"
    return "deviation"


def build_entry(atom_data: dict, atom_def: dict) -> dict:
    """
    Build a Lexicon Entry from expanded atom data.
    
    Returns the GPT-specified schema:
    {
      "atom": "EMO.like",
      "status": "proposed",
      "seed_synsets": [...],
      "core_pool": { "rules": [...], "count": N, "words": [...] },
      "deviation_pool": { "rules": [...], "count": N, "words": [...] },
      "deviation_stats": { ... },
      "meta": { ... }
    }
    """
    atom_id = atom_data["atom_id"]
    category = atom_data.get("category", "?")
    candidates = atom_data.get("candidates", [])
    
    # Classify each candidate
    core_words = []
    dev_words = []
    
    for c in candidates:
        pool = classify_candidate(c)
        
        # Determine primary source step
        primary_step = "unknown"
        for path in c.get("paths", []):
            if pool == "core" and path["step"] in CORE_STEPS:
                primary_step = path["step"]
                break
            elif pool == "deviation":
                primary_step = path["step"]
                break
        
        entry = {
            "w": c["lemma"],
            "pos": c["pos"],
            "src": primary_step,
            "definition": c.get("definition", ""),
            "path_count": len(c.get("paths", [])),
        }
        
        if pool == "core":
            # Record all core-qualifying sources
            core_sources = [p["step"] for p in c.get("paths", []) if p["step"] in CORE_STEPS]
            entry["sources"] = list(set(core_sources))
            core_words.append(entry)
        else:
            # Record all deviation sources
            dev_sources = [p["step"] for p in c.get("paths", [])]
            entry["sources"] = list(set(dev_sources))
            dev_words.append(entry)
    
    # Compute deviation stats
    total = len(candidates)
    core_count = len(core_words)
    dev_count = len(dev_words)
    
    # Source breakdown for deviation
    dev_source_counts = defaultdict(int)
    for w in dev_words:
        for s in w["sources"]:
            dev_source_counts[s] += 1
    
    # Step contribution from original expansion
    step_contributions = {}
    for s in atom_data.get("steps", []):
        step_contributions[s["step"]] = s["added"]
    
    entry = {
        "atom": atom_id,
        "category": category,
        "status": "proposed",
        "symmetric_pair": atom_def.get("symmetric_pair", ""),
        "definition_en": atom_def.get("definition_en", ""),
        "seed_synsets": [
            s["synset_id"] for s in 
            (json.loads(atom_data.get("_raw_seeds", "[]")) 
             if isinstance(atom_data.get("_raw_seeds"), str)
             else atom_data.get("_raw_seeds", []))
        ] if "_raw_seeds" in atom_data else [],
        "core_pool": {
            "rules": sorted(CORE_STEPS),
            "count": core_count,
            "words": sorted(core_words, key=lambda x: x["w"].lower()),
        },
        "deviation_pool": {
            "rules": sorted(DEVIATION_STEPS),
            "count": dev_count,
            "words": sorted(dev_words, key=lambda x: x["w"].lower()),
        },
        "deviation_stats": {
            "total_candidates": total,
            "core_count": core_count,
            "dev_count": dev_count,
            "core_ratio": round(core_count / total, 3) if total > 0 else 0,
            "dev_ratio": round(dev_count / total, 3) if total > 0 else 0,
            "dev_source_breakdown": dict(dev_source_counts),
            "step_contributions": step_contributions,
        },
        "meta": {
            "generator": "wn_lexicon_entry.py",
            "version": "1.0.0",
            "design": "3AI consolidated: Core/Deviation pool separation",
        },
    }
    
    return entry


def load_expanded(expand_dir: str) -> dict:
    """Load all individual atom expand results."""
    results = {}
    for f in sorted(Path(expand_dir).glob("*.json")):
        if f.name.startswith("_"):
            continue
        with open(f) as fh:
            data = json.load(fh)
        if "atom_id" in data:
            results[data["atom_id"]] = data
    return results


def load_dictionary(dict_path: str) -> dict:
    with open(dict_path) as f:
        raw = json.load(f)
    if "concepts" in raw and isinstance(raw["concepts"], dict):
        return raw["concepts"]
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory with expanded atom JSONs")
    parser.add_argument("--dictionary", required=True, help="esde_dictionary.json")
    parser.add_argument("--outdir", default="lexicon", help="Output directory for lexicon entries")
    parser.add_argument("--atoms", default=None, help="Comma-separated atom IDs (default: all)")
    args = parser.parse_args()
    
    results = load_expanded(args.dir)
    dictionary = load_dictionary(args.dictionary)
    
    if args.atoms:
        atom_filter = set(args.atoms.split(","))
        results = {k: v for k, v in results.items() if k in atom_filter}
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"Building lexicon entries for {len(results)} atoms...")
    print(f"Core steps: {sorted(CORE_STEPS)}")
    print(f"Deviation steps: {sorted(DEVIATION_STEPS)}")
    print()
    
    # Track aggregate stats
    all_entries = {}
    total_core = 0
    total_dev = 0
    zero_core = 0
    
    for i, (atom_id, data) in enumerate(sorted(results.items()), 1):
        atom_def = dictionary.get(atom_id, {})
        entry = build_entry(data, atom_def)
        all_entries[atom_id] = entry
        
        cc = entry["core_pool"]["count"]
        dc = entry["deviation_pool"]["count"]
        total_core += cc
        total_dev += dc
        if cc == 0:
            zero_core += 1
        
        status = "✅" if cc > 0 else "⚠️"
        print(f"  [{i:3d}/{len(results)}] {atom_id:25s}  core={cc:4d}  dev={dc:4d}  {status}")
        
        # Save individual entry
        fname = atom_id.replace(".", "_") + ".json"
        with open(outdir / fname, 'w') as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)
    
    # Save summary
    summary = {
        "meta": {
            "total_atoms": len(all_entries),
            "total_core_words": total_core,
            "total_dev_words": total_dev,
            "zero_core_atoms": zero_core,
            "avg_core_per_atom": round(total_core / len(all_entries), 1) if all_entries else 0,
            "avg_dev_per_atom": round(total_dev / len(all_entries), 1) if all_entries else 0,
            "core_steps": sorted(CORE_STEPS),
            "deviation_steps": sorted(DEVIATION_STEPS),
        },
        "atoms": {
            aid: {
                "core_count": e["core_pool"]["count"],
                "dev_count": e["deviation_pool"]["count"],
                "core_ratio": e["deviation_stats"]["core_ratio"],
                "status": e["status"],
            }
            for aid, e in sorted(all_entries.items())
        }
    }
    
    with open(outdir / "_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Print report
    print(f"\n{'='*70}")
    print(f"  LEXICON ENTRY GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Atoms: {len(all_entries)}")
    print(f"  Total Core words: {total_core}  (avg {total_core/len(all_entries):.1f}/atom)")
    print(f"  Total Deviation words: {total_dev}  (avg {total_dev/len(all_entries):.1f}/atom)")
    print(f"  Zero-core atoms: {zero_core}")
    print(f"  Core/Total ratio: {total_core/(total_core+total_dev)*100:.1f}%")
    
    # Category breakdown
    by_cat = defaultdict(list)
    for aid, e in all_entries.items():
        by_cat[e["category"]].append(e)
    
    print(f"\n  By category:")
    for cat, entries in sorted(by_cat.items()):
        cores = [e["core_pool"]["count"] for e in entries]
        devs = [e["deviation_pool"]["count"] for e in entries]
        import statistics
        med_c = statistics.median(cores)
        med_d = statistics.median(devs)
        z = sum(1 for c in cores if c == 0)
        print(f"    {cat:6s}: {len(entries):3d} atoms  core_med={med_c:5.0f}  dev_med={med_d:5.0f}  zero_core={z}")
    
    print(f"\n  ✅ Lexicon entries: {outdir}/")
    print(f"  ✅ Summary: {outdir}/_summary.json")


if __name__ == "__main__":
    main()
