#!/usr/bin/env python3
"""
Phase B Step B2: WordNet Evidence Builder for Lexicon
Validates QwQ proposals with WordNet data and expands with synonyms.

Usage:
    python wordnet_evidence_builder.py \
        --candidates ../output/qwq_candidates_EMO.like.json \
        --output ../output/evidence_EMO.like.json
"""

import argparse
import json
import re
from pathlib import Path

try:
    from nltk.corpus import wordnet as wn
    import nltk
    # Ensure wordnet data is available
    try:
        wn.synsets("test")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
except ImportError:
    print("ERROR: nltk is required. Install with: pip install nltk")
    raise


# ── POS Normalization (Final Patch A) ──────────────────────────

# Internal canonical: n/v/adj/adv
# WordNet uses: n/v/a/r/s
INTERNAL_TO_WN = {"n": "n", "v": "v", "adj": "a", "adv": "r"}
WN_TO_INTERNAL = {"n": "n", "v": "v", "a": "adj", "s": "adj", "r": "adv"}


def wn_pos_to_internal(wn_pos: str) -> str:
    return WN_TO_INTERNAL.get(wn_pos, wn_pos)


def internal_pos_to_wn(pos: str) -> str:
    return INTERNAL_TO_WN.get(pos, None)


# ── Word Normalization (Final Patch C) ─────────────────────────

def normalize_word(w: str) -> str:
    w = w.strip().lower()
    w = w.replace("_", " ")
    w = re.sub(r'\s+', ' ', w)
    return w


# ── Category Markers (GPT Patch #5) ───────────────────────────

STRICT_CATEGORY_MARKERS = {
    "EMO": ["emotion", "feeling", "affection", "sentiment", "passion", "state"],
}


# ── Axis WordNet Reliability ───────────────────────────────────

AXIS_WORDNET_RELIABILITY = {
    "temporal":         "medium-low",
    "scale":            "medium",
    "epistemological":  "medium",
    "ontological":      "medium-low",
    "interconnection":  "high",
    "resonance":        "low-medium",
    "symmetry":         "medium",
    "lawfulness":       "low",
    "experience":       "low",
    "value_generation": "low-medium",
}


# ── Evidence Builder ───────────────────────────────────────────

def build_evidence(word: str, pos: str, atom_category: str) -> dict:
    """Build WordNet evidence for a single word+pos."""

    evidence = {
        "synsets": [],
        "hypernym_chain": [],
        "antonyms": [],
        "entailments": [],
        "causes": [],
        "verb_groups": [],
        "definition": "",
        "category_compatible": None,
        "synonym_expansion": [],
        "wordnet_found": False,
    }

    # Convert to WordNet POS for targeted lookup
    wn_pos = internal_pos_to_wn(pos)
    wn_word = word.replace(" ", "_")  # WordNet uses underscores

    if wn_pos:
        synsets = wn.synsets(wn_word, pos=wn_pos)
    else:
        synsets = wn.synsets(wn_word)

    if not synsets:
        # Fallback: try without POS filter
        synsets = wn.synsets(wn_word)

    if not synsets:
        return evidence

    evidence["wordnet_found"] = True

    # Primary synset
    primary = synsets[0]
    evidence["synsets"] = [s.name() for s in synsets]
    evidence["definition"] = primary.definition()

    # Hypernym chain
    if primary.hypernym_paths():
        chain = [s.name().split('.')[0] for s in primary.hypernym_paths()[0]]
        evidence["hypernym_chain"] = chain

    # Category compatibility (GPT Patch #5)
    markers = STRICT_CATEGORY_MARKERS.get(atom_category)
    if markers:
        chain_str = " ".join(evidence["hypernym_chain"]).lower()
        evidence["category_compatible"] = any(m in chain_str for m in markers)
    else:
        evidence["category_compatible"] = None

    # Antonyms
    ant_set = set()
    for syn in synsets:
        for lemma in syn.lemmas():
            for ant in lemma.antonyms():
                ant_set.add(ant.name().replace("_", " "))
    evidence["antonyms"] = sorted(ant_set)

    # Entailments & causes (verbs)
    for syn in synsets:
        evidence["entailments"].extend([e.name() for e in syn.entailments()])
        evidence["causes"].extend([c.name() for c in syn.causes()])
        evidence["verb_groups"].extend([v.name() for v in syn.verb_groups()])

    # Synonym expansion (GPT Patch #6: always proposed)
    seen = {word}
    for syn in synsets[:3]:
        for lemma in syn.lemmas():
            name = normalize_word(lemma.name())
            syn_pos = wn_pos_to_internal(syn.pos())
            if name not in seen:
                evidence["synonym_expansion"].append({
                    "w": name,
                    "pos": syn_pos,
                    "status": "proposed",
                    "source_synset": syn.name(),
                })
                seen.add(name)

    return evidence


# ── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build WordNet evidence for Lexicon candidates")
    parser.add_argument("--candidates", required=True, help="QwQ candidates JSON")
    parser.add_argument("--output", required=True, help="Output evidence JSON")
    args = parser.parse_args()

    with open(args.candidates) as f:
        candidates = json.load(f)

    atom_id = candidates["atom"]
    atom_category = atom_id.split(".")[0]
    print(f"Building evidence for {atom_id} (category: {atom_category})")

    result = {
        "atom": atom_id,
        "category": atom_category,
        "slots": {},
        "stats": {
            "total_words": 0,
            "wordnet_found": 0,
            "wordnet_missing": 0,
            "category_compatible": 0,
            "category_mismatch": 0,
            "category_unknown": 0,
            "na_slots": 0,
        },
    }

    for slot_key, slot_data in candidates["slots"].items():
        axis = slot_key.split(".")[0]
        reliability = AXIS_WORDNET_RELIABILITY.get(axis, "unknown")

        if slot_data.get("na", False):
            result["slots"][slot_key] = {
                "na": True,
                "na_reason": slot_data.get("na_reason", ""),
                "words": [],
                "evidence": {},
            }
            result["stats"]["na_slots"] += 1
            continue

        slot_result = {
            "na": False,
            "axis_wordnet_reliability": reliability,
            "words": slot_data["words"],  # Preserve original candidates
            "evidence": {},
        }

        for word_entry in slot_data["words"]:
            w = word_entry["w"]
            pos = word_entry["pos"]
            key = f"{w}::{pos}"

            result["stats"]["total_words"] += 1

            ev = build_evidence(w, pos, atom_category)
            slot_result["evidence"][key] = ev

            if ev["wordnet_found"]:
                result["stats"]["wordnet_found"] += 1
            else:
                result["stats"]["wordnet_missing"] += 1

            if ev["category_compatible"] is True:
                result["stats"]["category_compatible"] += 1
            elif ev["category_compatible"] is False:
                result["stats"]["category_mismatch"] += 1
            else:
                result["stats"]["category_unknown"] += 1

        result["slots"][slot_key] = slot_result

    # Print stats
    s = result["stats"]
    print(f"\nResults:")
    print(f"  Total words:         {s['total_words']}")
    print(f"  WordNet found:       {s['wordnet_found']}")
    print(f"  WordNet missing:     {s['wordnet_missing']}")
    print(f"  Category compatible: {s['category_compatible']}")
    print(f"  Category mismatch:   {s['category_mismatch']}")
    print(f"  Category unknown:    {s['category_unknown']}")
    print(f"  N/A slots:           {s['na_slots']}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
