#!/usr/bin/env python3
"""
ESDE Lexicon v2 — Max Expansion Test
=====================================
Try ALL WordNet relations at maximum reasonable depth.
Report each step with counts. Then we trim.

Usage:
  python3 wn_max_expand.py EMO.like
  python3 wn_max_expand.py EMO.anger
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

try:
    from nltk.corpus import wordnet as wn
    wn.synsets('test')
except:
    print("❌ NLTK WordNet not available. Run:")
    print("   pip install nltk")
    print("   python3 -c \"import nltk; nltk.download('wordnet')\"")
    sys.exit(1)


# ════════════════════════════════════════════════════════════════
#  SEED DEFINITIONS (same as wn_lexicon.py)
# ════════════════════════════════════════════════════════════════

ATOM_SEEDS = {
    "EMO.like": {
        "kanji": "好",
        "kanji_field": "favorable regard, fondness, affinity, to be fond of, pleasing",
        "seeds": ["like.v.02", "fondness.n.01", "affection.n.01", "likable.a.01"],
        "exclude": {"similar", "alike", "such_as", "prefer", "desire", "love",
                    "entity", "object", "thing", "person", "people"},
    },
    "EMO.anger": {
        "kanji": "怒",
        "kanji_field": "anger, wrath, rage, indignation, to be angry",
        "seeds": ["anger.n.01", "anger.n.02", "angry.a.01", "rage.n.01"],
        "exclude": {"hate", "hatred", "contempt", "fear",
                    "entity", "object", "thing", "person", "people"},
    },
}


def pos_label(p):
    return {"n": "n", "v": "v", "a": "adj", "s": "adj", "r": "adv"}.get(p, p)


def collect_lemmas(synset):
    """Get all lemma strings from a synset."""
    return [l.name().replace('_', ' ') for l in synset.lemmas()]


def expand_max(atom_id: str):
    """Maximum expansion with per-step reporting."""
    
    cfg = ATOM_SEEDS[atom_id]
    exclude = {x.lower() for x in cfg["exclude"]}
    
    print(f"\n{'='*70}")
    print(f"  MAX EXPAND: {atom_id}")
    print(f"  Kanji: {cfg['kanji']} ({cfg['kanji_field']})")
    print(f"  Seeds: {cfg['seeds']}")
    print(f"{'='*70}")
    
    # Master collection: (lemma_lower, pos) → {paths[], definition}
    all_candidates = {}
    step_log = []
    step_words = defaultdict(list)  # step → list of (lemma, pos) added at this step
    
    def add(lemma, pos, source_step, from_synset, via_synset, relation, definition):
        key = (lemma.lower().strip(), pos)
        if key[0] in exclude:
            return False
        if len(key[0]) <= 1:
            return False
        is_new = key not in all_candidates
        if is_new:
            all_candidates[key] = {
                "lemma": lemma, "pos": pos,
                "paths": [], "definition": definition,
            }
            step_words[source_step].append(f"{lemma}({pos})")
        path = {
            "step": source_step,
            "relation": relation,
            "from_synset": from_synset,
            "via_synset": via_synset,
        }
        if path not in all_candidates[key]["paths"]:
            all_candidates[key]["paths"].append(path)
        return is_new

    def print_step(step_id, desc, added, total):
        """Print step summary + the actual words added."""
        step_log.append((step_id, desc, added))
        print(f"  {step_id:22s} → +{added:3d}  (total: {total})  {desc}")
        words = step_words.get(step_id, [])
        if words:
            line = "          "
            for w in words:
                if len(line) + len(w) + 2 > 90:
                    print(line.rstrip(", "))
                    line = "          "
                line += w + ", "
            if line.strip():
                print(line.rstrip(", "))
    
    # ── STEP 0: Seed synsets themselves ──
    before = len(all_candidates)
    seed_synsets = []
    for sid in cfg["seeds"]:
        try:
            ss = wn.synset(sid)
            seed_synsets.append(ss)
            for lemma in collect_lemmas(ss):
                add(lemma, pos_label(ss.pos()), "0_seed", sid, ss.name(), "seed_lemma", ss.definition())
        except Exception as e:
            print(f"  ⚠️ Seed {sid} not found: {e}")
    added = len(all_candidates) - before
    print_step("0_seed", f"Seed synset lemmas ({len(cfg['seeds'])} synsets)", added, len(all_candidates))
    
    # ── STEP 1: Synonyms in same synset (already done in step 0) ──
    # Covered by step 0
    
    # ── STEP 2: Hypernyms (1 level) ──
    before = len(all_candidates)
    for ss in seed_synsets:
        for hyper in ss.hypernyms():
            for lemma in collect_lemmas(hyper):
                add(lemma, pos_label(hyper.pos()), "2_hypernym_d1", ss.name(), hyper.name(), "hypernym", hyper.definition())
    added = len(all_candidates) - before
    print_step("2_hypernym_d1", "Hypernyms (depth 1)", added, len(all_candidates))
    
    # ── STEP 3: Hyponyms (depth 1) ──
    before = len(all_candidates)
    hypo_d1 = []
    for ss in seed_synsets:
        for hypo in ss.hyponyms():
            hypo_d1.append(hypo)
            for lemma in collect_lemmas(hypo):
                add(lemma, pos_label(hypo.pos()), "3_hyponym_d1", ss.name(), hypo.name(), "hyponym", hypo.definition())
    added = len(all_candidates) - before
    print_step("3_hyponym_d1", f"Hyponyms depth 1 ({len(hypo_d1)} synsets)", added, len(all_candidates))
    
    # ── STEP 4: Hyponyms (depth 2) ──
    before = len(all_candidates)
    hypo_d2 = []
    for h1 in hypo_d1:
        for hypo in h1.hyponyms():
            hypo_d2.append(hypo)
            for lemma in collect_lemmas(hypo):
                add(lemma, pos_label(hypo.pos()), "4_hyponym_d2", h1.name(), hypo.name(), "hyponym", hypo.definition())
    added = len(all_candidates) - before
    print_step("4_hyponym_d2", f"Hyponyms depth 2 ({len(hypo_d2)} synsets)", added, len(all_candidates))
    
    # ── STEP 5: Hyponyms (depth 3) ──
    before = len(all_candidates)
    hypo_d3 = []
    for h2 in hypo_d2:
        for hypo in h2.hyponyms():
            hypo_d3.append(hypo)
            for lemma in collect_lemmas(hypo):
                add(lemma, pos_label(hypo.pos()), "5_hyponym_d3", h2.name(), hypo.name(), "hyponym", hypo.definition())
    added = len(all_candidates) - before
    print_step("5_hyponym_d3", f"Hyponyms depth 3 ({len(hypo_d3)} synsets)", added, len(all_candidates))
    
    # ── STEP 6: Derivationally related forms (from ALL collected synsets) ──
    before = len(all_candidates)
    all_synsets_so_far = set()
    for ss in seed_synsets + hypo_d1 + hypo_d2:
        all_synsets_so_far.add(ss)
    
    deriv_count = 0
    for ss in all_synsets_so_far:
        for lemma_obj in ss.lemmas():
            for related in lemma_obj.derivationally_related_forms():
                rel_ss = related.synset()
                name = related.name().replace('_', ' ')
                if add(name, pos_label(rel_ss.pos()), "6_derivational", ss.name(), rel_ss.name(), "derivational", rel_ss.definition()):
                    deriv_count += 1
    added = len(all_candidates) - before
    print_step("6_derivational", f"Derivationally related (from {len(all_synsets_so_far)} synsets)", added, len(all_candidates))
    
    # ── STEP 7: Similar_to (adjective satellites) ──
    before = len(all_candidates)
    sim_count = 0
    for ss in all_synsets_so_far:
        for sim in ss.similar_tos():
            for lemma in collect_lemmas(sim):
                add(lemma, pos_label(sim.pos()), "7_similar_to", ss.name(), sim.name(), "similar_to", sim.definition())
            sim_count += 1
    added = len(all_candidates) - before
    print_step("7_similar_to", f"Similar_to ({sim_count} synsets visited)", added, len(all_candidates))
    
    # ── STEP 8: Also_see ──
    before = len(all_candidates)
    for ss in all_synsets_so_far:
        for also in ss.also_sees():
            for lemma in collect_lemmas(also):
                add(lemma, pos_label(also.pos()), "8_also_see", ss.name(), also.name(), "also_see", also.definition())
    added = len(all_candidates) - before
    print_step("8_also_see", "Also_see", added, len(all_candidates))
    
    # ── STEP 9: Antonyms ──
    before = len(all_candidates)
    for ss in seed_synsets:
        for lemma_obj in ss.lemmas():
            for ant in lemma_obj.antonyms():
                ant_ss = ant.synset()
                name = ant.name().replace('_', ' ')
                add(name, pos_label(ant_ss.pos()), "9_antonym", ss.name(), ant_ss.name(), "antonym", ant_ss.definition())
    added = len(all_candidates) - before
    print_step("9_antonym", "Antonyms (seed only)", added, len(all_candidates))
    
    # ── STEP 10: Sibling synsets (same hypernym as seeds) ──
    before = len(all_candidates)
    sibling_count = 0
    for ss in seed_synsets:
        for hyper in ss.hypernyms():
            for sibling in hyper.hyponyms():
                if sibling == ss:
                    continue
                sibling_count += 1
                for lemma in collect_lemmas(sibling):
                    add(lemma, pos_label(sibling.pos()), "10_sibling", hyper.name(), sibling.name(), "sibling", sibling.definition())
    added = len(all_candidates) - before
    print_step("10_sibling", f"Siblings ({sibling_count} synsets)", added, len(all_candidates))
    
    # ── STEP 11: Pertainyms (adj → noun they pertain to) ──
    before = len(all_candidates)
    for ss in all_synsets_so_far:
        if ss.pos() in ('a', 's'):
            for lemma_obj in ss.lemmas():
                for pert in lemma_obj.pertainyms():
                    pert_ss = pert.synset()
                    name = pert.name().replace('_', ' ')
                    add(name, pos_label(pert_ss.pos()), "11_pertainym", ss.name(), pert_ss.name(), "pertainym", pert_ss.definition())
    added = len(all_candidates) - before
    print_step("11_pertainym", "Pertainyms", added, len(all_candidates))
    
    # ── STEP 12: Verb groups ──
    before = len(all_candidates)
    for ss in all_synsets_so_far:
        if ss.pos() == 'v':
            for vg in ss.verb_groups():
                for lemma in collect_lemmas(vg):
                    add(lemma, pos_label(vg.pos()), "12_verb_group", ss.name(), vg.name(), "verb_group", vg.definition())
    added = len(all_candidates) - before
    print_step("12_verb_group", "Verb groups", added, len(all_candidates))
    
    # ══════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════
    
    print(f"\n{'─'*70}")
    print(f"  EXPANSION SUMMARY")
    print(f"{'─'*70}")
    
    total = len(all_candidates)
    print(f"\n  Total unique candidates: {total}")
    
    # POS breakdown
    pos_counts = defaultdict(int)
    for (lemma, pos) in all_candidates:
        pos_counts[pos] += 1
    print(f"  POS: {dict(pos_counts)}")
    
    # Step contribution
    print(f"\n  Step-by-step contribution:")
    cumulative = 0
    for step_id, desc, count in step_log:
        cumulative += count
        bar = '█' * min(count, 50)
        pct = count / total * 100 if total > 0 else 0
        print(f"    {step_id:20s}  +{count:3d} ({pct:4.1f}%)  {bar}  {desc}")
    
    # Source distribution
    print(f"\n  Source distribution (words may have multiple paths):")
    source_counts = defaultdict(int)
    for v in all_candidates.values():
        for p in v["paths"]:
            source_counts[p["step"]] += 1
    for src, cnt in sorted(source_counts.items()):
        print(f"    {src:20s}  {cnt:3d}")
    
    # Multi-path words (found through 2+ routes)
    multi = [(k, v) for k, v in all_candidates.items() if len(v["paths"]) > 1]
    if multi:
        print(f"\n  Multi-path words ({len(multi)} words found via 2+ routes):")
        for (lemma, pos), v in sorted(multi, key=lambda x: -len(x[1]["paths"]))[:15]:
            routes = [f"{p['relation']}←{p['from_synset']}" for p in v["paths"]]
            print(f"    {v['lemma']:25s} {pos:4s}  {len(v['paths'])} paths: {'; '.join(routes)}")
    
    # Sample words by source
    print(f"\n  Sample words by deepest step:")
    by_step = defaultdict(list)
    for (lemma, pos), v in all_candidates.items():
        deepest = max(v["paths"], key=lambda p: int(p["step"].split('_')[0]))
        by_step[deepest["step"]].append(lemma)
    
    for step in sorted(by_step.keys()):
        words = by_step[step][:8]
        print(f"    {step:20s}  {', '.join(words)}")
    
    # ── Save full list ──
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    
    output = {
        "atom": atom_id,
        "total_candidates": total,
        "pos_counts": dict(pos_counts),
        "steps": [{"step": s, "description": d, "added": c} for s, d, c in step_log],
        "candidates": [
            {
                "lemma": v["lemma"],
                "pos": v["pos"],
                "paths": v["paths"],
                "definition": v["definition"],
            }
            for (lemma, pos), v in sorted(all_candidates.items())
        ]
    }
    
    out_path = out_dir / f"{atom_id.replace('.','_')}_max_expand.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ Saved: {out_path}")
    
    # ── Full word list (for review) ──
    print(f"\n{'─'*70}")
    print(f"  FULL CANDIDATE LIST ({total} words)")
    print(f"{'─'*70}")
    for i, ((lemma, pos), v) in enumerate(sorted(all_candidates.items()), 1):
        steps = ','.join(sorted(set(p["step"].split('_')[0] for p in v["paths"])))
        n_paths = len(v["paths"])
        path_mark = f"×{n_paths}" if n_paths > 1 else "    "
        print(f"  {i:3d}. {v['lemma']:30s} {pos:4s}  [{steps:10s}] {path_mark} {v['definition'][:50]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 wn_max_expand.py EMO.like")
        sys.exit(1)
    expand_max(sys.argv[1])
