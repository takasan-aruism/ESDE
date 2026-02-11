#!/usr/bin/env python3
"""
ESDE Lexicon v2 — Auto Seed Generator
======================================
Reads esde_dictionary.json → generates seed synsets for all 326 atoms.

Strategy per atom:
  1. Look up atom.name in WordNet → take top synsets matching definition
  2. Look up each trigger word → take synsets in same semantic neighborhood
  3. Use anti_triggers as exclude list
  4. Output: seeds.json with per-atom seed definitions

Usage:
  python3 wn_auto_seed.py --dictionary esde_dictionary.json --out seeds.json
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict

try:
    from nltk.corpus import wordnet as wn
    wn.synsets('test')
    WN_AVAILABLE = True
except:
    WN_AVAILABLE = False
    print("⚠️ NLTK WordNet not available. Install with:")
    print("   pip install nltk")
    print("   python3 -c \"import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')\"")


def pos_label(p):
    return {"n": "n", "v": "v", "a": "adj", "s": "adj", "r": "adv"}.get(p, p)


def find_best_synsets(word, definition_hint="", max_synsets=4):
    """Find WordNet synsets for a word, optionally ranked by definition similarity."""
    if not WN_AVAILABLE:
        return []
    
    synsets = wn.synsets(word.replace(' ', '_'))
    if not synsets:
        return []
    
    # Simple ranking: prefer synsets whose definition overlaps with our definition
    if definition_hint:
        hint_words = set(definition_hint.lower().split())
        scored = []
        for ss in synsets:
            defn_words = set(ss.definition().lower().split())
            overlap = len(hint_words & defn_words)
            scored.append((overlap, ss))
        scored.sort(key=lambda x: -x[0])
        synsets = [s for _, s in scored]
    
    return synsets[:max_synsets]


def generate_seeds(dictionary_path: str, output_path: str):
    """Generate seed definitions for all atoms."""
    
    with open(dictionary_path) as f:
        raw = json.load(f)
    
    # Support both flat dict and nested {"concepts": {...}} format
    if "concepts" in raw and isinstance(raw["concepts"], dict):
        dictionary = raw["concepts"]
    else:
        dictionary = raw
    
    print(f"Loaded {len(dictionary)} atoms from {dictionary_path}")
    
    # Category stats
    categories = defaultdict(int)
    for atom_id, atom in dictionary.items():
        categories[atom["category"]] += 1
    
    print(f"\nCategories ({len(categories)}):")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} atoms")
    
    seeds = {}
    stats = {
        "total_atoms": len(dictionary),
        "atoms_with_seeds": 0,
        "atoms_without_seeds": 0,
        "total_seed_synsets": 0,
        "by_category": {},
    }
    
    no_seed_atoms = []
    
    for atom_id, atom in sorted(dictionary.items()):
        name = atom["name"]
        category = atom["category"]
        definition = atom.get("definition_en", "")
        triggers = atom.get("triggers_en", [])
        anti_triggers = atom.get("anti_triggers_en", [])
        symmetric_pair = atom.get("symmetric_pair", "")
        
        # Build exclude list from anti_triggers + symmetric pair name
        exclude = set(t.lower() for t in anti_triggers)
        if symmetric_pair and symmetric_pair in dictionary:
            pair_name = dictionary[symmetric_pair]["name"]
            exclude.add(pair_name.lower())
            # Also add pair's triggers
            for t in dictionary[symmetric_pair].get("triggers_en", []):
                exclude.add(t.lower())
        
        # Find seed synsets
        # Strategy: name first, then triggers
        all_synsets = []
        
        # 1. From atom name
        name_synsets = find_best_synsets(name, definition, max_synsets=3)
        for ss in name_synsets:
            all_synsets.append({
                "synset_id": ss.name(),
                "lemma": name,
                "pos": pos_label(ss.pos()),
                "definition": ss.definition(),
                "source": "atom_name",
            })
        
        # 2. From trigger words (only if they add new synsets)
        seen_synset_ids = {ss.name() for ss in name_synsets}
        for trigger in triggers[:5]:  # limit to first 5 triggers
            trigger_synsets = find_best_synsets(trigger, definition, max_synsets=2)
            for ss in trigger_synsets:
                if ss.name() not in seen_synset_ids:
                    seen_synset_ids.add(ss.name())
                    all_synsets.append({
                        "synset_id": ss.name(),
                        "lemma": trigger,
                        "pos": pos_label(ss.pos()),
                        "definition": ss.definition(),
                        "source": "trigger",
                    })
        
        # Limit total seeds per atom
        all_synsets = all_synsets[:6]
        
        # Record
        seeds[atom_id] = {
            "atom_id": atom_id,
            "category": category,
            "name": name,
            "definition_en": definition,
            "symmetric_pair": symmetric_pair,
            "seed_synsets": all_synsets,
            "seed_count": len(all_synsets),
            "exclude_lemmas": sorted(exclude),
        }
        
        if all_synsets:
            stats["atoms_with_seeds"] += 1
            stats["total_seed_synsets"] += len(all_synsets)
        else:
            stats["atoms_without_seeds"] += 1
            no_seed_atoms.append(atom_id)
        
        # Category stats
        if category not in stats["by_category"]:
            stats["by_category"][category] = {"total": 0, "with_seeds": 0, "avg_seeds": 0}
        stats["by_category"][category]["total"] += 1
        if all_synsets:
            stats["by_category"][category]["with_seeds"] += 1
    
    # Compute averages
    for cat, cat_stats in stats["by_category"].items():
        cat_atoms = [s for s in seeds.values() if s["category"] == cat]
        cat_seed_counts = [s["seed_count"] for s in cat_atoms]
        cat_stats["avg_seeds"] = sum(cat_seed_counts) / len(cat_seed_counts) if cat_seed_counts else 0
    
    # Save
    output = {
        "stats": stats,
        "seeds": seeds,
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # Report
    print(f"\n{'='*60}")
    print(f"  SEED GENERATION REPORT")
    print(f"{'='*60}")
    print(f"  Total atoms:      {stats['total_atoms']}")
    print(f"  With seeds:       {stats['atoms_with_seeds']}")
    print(f"  Without seeds:    {stats['atoms_without_seeds']}")
    print(f"  Total synsets:    {stats['total_seed_synsets']}")
    print(f"  Avg seeds/atom:   {stats['total_seed_synsets']/stats['total_atoms']:.1f}")
    
    print(f"\n  By category:")
    for cat, cs in sorted(stats["by_category"].items()):
        print(f"    {cat:6s}: {cs['with_seeds']:3d}/{cs['total']:3d} atoms with seeds  (avg {cs['avg_seeds']:.1f} seeds/atom)")
    
    if no_seed_atoms:
        print(f"\n  Atoms without seeds ({len(no_seed_atoms)}):")
        for a in no_seed_atoms:
            print(f"    {a}")
    
    print(f"\n  ✅ Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary", default="esde_dictionary.json", help="Path to esde_dictionary.json")
    parser.add_argument("--out", default="seeds.json", help="Output path")
    args = parser.parse_args()
    
    if not WN_AVAILABLE:
        sys.exit(1)
    
    generate_seeds(args.dictionary, args.out)