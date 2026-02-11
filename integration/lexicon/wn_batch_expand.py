#!/usr/bin/env python3
"""
ESDE Lexicon v2 — Batch Expand All Atoms
=========================================
Reads seeds.json → runs WordNet expansion for all 326 atoms → saves per-atom results.

Usage:
  python3 wn_batch_expand.py --seeds seeds.json --outdir expanded/
  python3 wn_batch_expand.py --seeds seeds.json --outdir expanded/ --steps 0,2,3,4,6,7,9
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict

try:
    from nltk.corpus import wordnet as wn
    wn.synsets('test')
except:
    print("❌ NLTK WordNet required")
    sys.exit(1)


def pos_label(p):
    return {"n": "n", "v": "v", "a": "adj", "s": "adj", "r": "adv"}.get(p, p)


def collect_lemmas(synset):
    return [l.name().replace('_', ' ') for l in synset.lemmas()]


def expand_atom(atom_seed: dict, enabled_steps: set = None) -> dict:
    """
    Expand a single atom. Returns structured result with per-step tracking.
    
    enabled_steps: set of step numbers to run (default: all)
    """
    atom_id = atom_seed["atom_id"]
    exclude = set(x.lower() for x in atom_seed.get("exclude_lemmas", []))
    # Also exclude very generic words
    exclude.update({"entity", "object", "thing", "person", "people", "something"})
    
    if enabled_steps is None:
        enabled_steps = {0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
    
    all_candidates = {}
    step_log = []
    step_words = defaultdict(list)
    
    def add(lemma, pos, step_id, from_synset, via_synset, relation, definition):
        key = (lemma.lower().strip(), pos)
        if key[0] in exclude or len(key[0]) <= 1:
            return False
        is_new = key not in all_candidates
        if is_new:
            all_candidates[key] = {
                "lemma": lemma, "pos": pos,
                "paths": [], "definition": definition,
            }
            step_words[step_id].append(f"{lemma}({pos})")
        path = {"step": step_id, "relation": relation,
                "from_synset": from_synset, "via_synset": via_synset}
        if path not in all_candidates[key]["paths"]:
            all_candidates[key]["paths"].append(path)
        return is_new
    
    # Load seed synsets
    seed_synsets = []
    before = len(all_candidates)
    if 0 in enabled_steps:
        for s in atom_seed.get("seed_synsets", []):
            try:
                ss = wn.synset(s["synset_id"])
                seed_synsets.append(ss)
                for lemma in collect_lemmas(ss):
                    add(lemma, pos_label(ss.pos()), "0_seed",
                        s["synset_id"], ss.name(), "seed_lemma", ss.definition())
            except:
                pass
    added = len(all_candidates) - before
    step_log.append(("0_seed", added, list(step_words.get("0_seed", []))))
    
    # Step 2: Hypernyms
    before = len(all_candidates)
    if 2 in enabled_steps:
        for ss in seed_synsets:
            for hyper in ss.hypernyms():
                for lemma in collect_lemmas(hyper):
                    add(lemma, pos_label(hyper.pos()), "2_hypernym_d1",
                        ss.name(), hyper.name(), "hypernym", hyper.definition())
    added = len(all_candidates) - before
    step_log.append(("2_hypernym_d1", added, list(step_words.get("2_hypernym_d1", []))))
    
    # Step 3: Hyponyms d1
    before = len(all_candidates)
    hypo_d1 = []
    if 3 in enabled_steps:
        for ss in seed_synsets:
            for hypo in ss.hyponyms():
                hypo_d1.append(hypo)
                for lemma in collect_lemmas(hypo):
                    add(lemma, pos_label(hypo.pos()), "3_hyponym_d1",
                        ss.name(), hypo.name(), "hyponym", hypo.definition())
    added = len(all_candidates) - before
    step_log.append(("3_hyponym_d1", added, list(step_words.get("3_hyponym_d1", []))))
    
    # Step 4: Hyponyms d2
    before = len(all_candidates)
    hypo_d2 = []
    if 4 in enabled_steps:
        for h1 in hypo_d1:
            for hypo in h1.hyponyms():
                hypo_d2.append(hypo)
                for lemma in collect_lemmas(hypo):
                    add(lemma, pos_label(hypo.pos()), "4_hyponym_d2",
                        h1.name(), hypo.name(), "hyponym", hypo.definition())
    added = len(all_candidates) - before
    step_log.append(("4_hyponym_d2", added, list(step_words.get("4_hyponym_d2", []))))
    
    # Step 5: Hyponyms d3
    before = len(all_candidates)
    if 5 in enabled_steps:
        for h2 in hypo_d2:
            for hypo in h2.hyponyms():
                for lemma in collect_lemmas(hypo):
                    add(lemma, pos_label(hypo.pos()), "5_hyponym_d3",
                        h2.name(), hypo.name(), "hyponym", hypo.definition())
    added = len(all_candidates) - before
    step_log.append(("5_hyponym_d3", added, list(step_words.get("5_hyponym_d3", []))))
    
    # Step 6: Derivationally related
    before = len(all_candidates)
    all_synsets_so_far = set(seed_synsets + hypo_d1 + hypo_d2)
    if 6 in enabled_steps:
        for ss in list(all_synsets_so_far):
            for lemma_obj in ss.lemmas():
                for related in lemma_obj.derivationally_related_forms():
                    rel_ss = related.synset()
                    name = related.name().replace('_', ' ')
                    add(name, pos_label(rel_ss.pos()), "6_derivational",
                        ss.name(), rel_ss.name(), "derivational", rel_ss.definition())
    added = len(all_candidates) - before
    step_log.append(("6_derivational", added, list(step_words.get("6_derivational", []))))
    
    # Step 7: Similar_to
    before = len(all_candidates)
    if 7 in enabled_steps:
        for ss in list(all_synsets_so_far):
            for sim in ss.similar_tos():
                for lemma in collect_lemmas(sim):
                    add(lemma, pos_label(sim.pos()), "7_similar_to",
                        ss.name(), sim.name(), "similar_to", sim.definition())
    added = len(all_candidates) - before
    step_log.append(("7_similar_to", added, list(step_words.get("7_similar_to", []))))
    
    # Step 8: Also_see
    before = len(all_candidates)
    if 8 in enabled_steps:
        for ss in list(all_synsets_so_far):
            for also in ss.also_sees():
                for lemma in collect_lemmas(also):
                    add(lemma, pos_label(also.pos()), "8_also_see",
                        ss.name(), also.name(), "also_see", also.definition())
    added = len(all_candidates) - before
    step_log.append(("8_also_see", added, list(step_words.get("8_also_see", []))))
    
    # Step 9: Antonyms
    before = len(all_candidates)
    if 9 in enabled_steps:
        for ss in seed_synsets:
            for lemma_obj in ss.lemmas():
                for ant in lemma_obj.antonyms():
                    ant_ss = ant.synset()
                    name = ant.name().replace('_', ' ')
                    add(name, pos_label(ant_ss.pos()), "9_antonym",
                        ss.name(), ant_ss.name(), "antonym", ant_ss.definition())
    added = len(all_candidates) - before
    step_log.append(("9_antonym", added, list(step_words.get("9_antonym", []))))
    
    # Step 10: Siblings
    before = len(all_candidates)
    if 10 in enabled_steps:
        for ss in seed_synsets:
            for hyper in ss.hypernyms():
                for sibling in hyper.hyponyms():
                    if sibling == ss:
                        continue
                    for lemma in collect_lemmas(sibling):
                        add(lemma, pos_label(sibling.pos()), "10_sibling",
                            hyper.name(), sibling.name(), "sibling", sibling.definition())
    added = len(all_candidates) - before
    step_log.append(("10_sibling", added, list(step_words.get("10_sibling", []))))
    
    # Step 11: Pertainyms
    before = len(all_candidates)
    if 11 in enabled_steps:
        for ss in list(all_synsets_so_far):
            if ss.pos() in ('a', 's'):
                for lemma_obj in ss.lemmas():
                    for pert in lemma_obj.pertainyms():
                        pert_ss = pert.synset()
                        name = pert.name().replace('_', ' ')
                        add(name, pos_label(pert_ss.pos()), "11_pertainym",
                            ss.name(), pert_ss.name(), "pertainym", pert_ss.definition())
    added = len(all_candidates) - before
    step_log.append(("11_pertainym", added, list(step_words.get("11_pertainym", []))))
    
    # Step 12: Verb groups
    before = len(all_candidates)
    if 12 in enabled_steps:
        for ss in list(all_synsets_so_far):
            if ss.pos() == 'v':
                for vg in ss.verb_groups():
                    for lemma in collect_lemmas(vg):
                        add(lemma, pos_label(vg.pos()), "12_verb_group",
                            ss.name(), vg.name(), "verb_group", vg.definition())
    added = len(all_candidates) - before
    step_log.append(("12_verb_group", added, list(step_words.get("12_verb_group", []))))
    
    # POS counts
    pos_counts = defaultdict(int)
    for (lemma, pos) in all_candidates:
        pos_counts[pos] += 1
    
    # Source counts
    source_counts = defaultdict(int)
    for v in all_candidates.values():
        for p in v["paths"]:
            source_counts[p["step"]] += 1
    
    return {
        "atom_id": atom_id,
        "category": atom_seed["category"],
        "seed_count": len(seed_synsets),
        "total_keys": len(all_candidates),
        "pos_counts": dict(pos_counts),
        "source_counts": dict(source_counts),
        "steps": [{"step": s, "added": a, "words": w} for s, a, w in step_log],
        "candidates": [
            {
                "lemma": v["lemma"], "pos": v["pos"],
                "paths": v["paths"], "definition": v["definition"],
            }
            for (lemma, pos), v in sorted(all_candidates.items())
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", required=True, help="Path to seeds.json")
    parser.add_argument("--outdir", default="expanded", help="Output directory")
    parser.add_argument("--steps", default=None, help="Comma-separated step numbers (default: all)")
    parser.add_argument("--atoms", default=None, help="Comma-separated atom IDs (default: all)")
    args = parser.parse_args()
    
    enabled_steps = None
    if args.steps:
        enabled_steps = set(int(x) for x in args.steps.split(","))
    
    with open(args.seeds) as f:
        data = json.load(f)
    
    all_seeds = data["seeds"]
    
    # Filter atoms if specified
    if args.atoms:
        atom_filter = set(args.atoms.split(","))
        all_seeds = {k: v for k, v in all_seeds.items() if k in atom_filter}
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"Expanding {len(all_seeds)} atoms...")
    print(f"Steps: {sorted(enabled_steps) if enabled_steps else 'all'}")
    print()
    
    results = {}
    for i, (atom_id, seed) in enumerate(sorted(all_seeds.items()), 1):
        result = expand_atom(seed, enabled_steps)
        results[atom_id] = result
        
        # Progress
        total = result["total_keys"]
        seeds = result["seed_count"]
        status = "✅" if total > 0 else "⚠️" if seeds == 0 else "❌"
        print(f"  [{i:3d}/{len(all_seeds)}] {atom_id:25s}  seeds={seeds}  total={total:4d}  {status}")
        
        # Save individual result
        atom_file = outdir / f"{atom_id.replace('.', '_')}.json"
        with open(atom_file, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    # Save combined results (candidates stripped for size)
    summary = {}
    for atom_id, r in results.items():
        summary[atom_id] = {
            "atom_id": r["atom_id"],
            "category": r["category"],
            "seed_count": r["seed_count"],
            "total_keys": r["total_keys"],
            "pos_counts": r["pos_counts"],
            "source_counts": r["source_counts"],
            "steps": r["steps"],
            # candidates omitted for size — see individual files
        }
    
    summary_path = outdir / "_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Print overall stats
    totals = [r["total_keys"] for r in results.values()]
    zeros = sum(1 for t in totals if t == 0)
    thin = sum(1 for t in totals if 0 < t < 15)
    
    print(f"\n{'='*60}")
    print(f"  BATCH EXPAND COMPLETE")
    print(f"{'='*60}")
    print(f"  Atoms processed: {len(results)}")
    print(f"  Total candidates: {sum(totals)}")
    print(f"  Avg per atom: {sum(totals)/len(totals):.1f}")
    print(f"  Min/Max: {min(totals)} / {max(totals)}")
    print(f"  Zero (no candidates): {zeros}")
    print(f"  Thin (<15 candidates): {thin}")
    print(f"\n  ✅ Summary: {summary_path}")
    print(f"  ✅ Individual files: {outdir}/")


if __name__ == "__main__":
    main()
