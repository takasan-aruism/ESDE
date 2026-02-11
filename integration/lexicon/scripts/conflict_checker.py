#!/usr/bin/env python3
"""
Phase B Step B3: Conflict Checker for Lexicon
Detects symmetric pair collisions, category mismatches, and multi-atom ambiguity.

Usage:
    python conflict_checker.py \
        --evidence ../output/evidence_EMO.like.json \
        --evidence-pair ../output/evidence_EMO.dislike.json \
        --atoms ../definitions/atoms_v1.json \
        --output ../output/conflicts_EMO.like.json
"""

import argparse
import json
from pathlib import Path


# ── B3a: Symmetric Pair Collision (GPT Patch #4: lemma::pos) ──

def check_symmetric_collision(evidence_a: dict, evidence_b: dict) -> list:
    """Flag lemma::pos appearing in both sides of a symmetric pair."""
    conflicts = []

    keys_a = set()
    keys_b = set()

    for slot_data in evidence_a["slots"].values():
        for w in slot_data.get("words", []):
            keys_a.add(f"{w['w']}::{w['pos']}")

    for slot_data in evidence_b["slots"].values():
        for w in slot_data.get("words", []):
            keys_b.add(f"{w['w']}::{w['pos']}")

    overlap = keys_a & keys_b
    if overlap:
        for key in sorted(overlap):
            # Find which slots they appear in
            slots_a = []
            slots_b = []
            for sk, sd in evidence_a["slots"].items():
                for w in sd.get("words", []):
                    if f"{w['w']}::{w['pos']}" == key:
                        slots_a.append(sk)
            for sk, sd in evidence_b["slots"].items():
                for w in sd.get("words", []):
                    if f"{w['w']}::{w['pos']}" == key:
                        slots_b.append(sk)

            conflicts.append({
                "type": "symmetric_collision",
                "key": key,
                "atom_a": evidence_a["atom"],
                "atom_b": evidence_b["atom"],
                "slots_a": slots_a,
                "slots_b": slots_b,
                "severity": "HIGH",
                "action": "Remove from one side. Word cannot serve both symmetric pair atoms.",
            })

    return conflicts


# ── B3b: Category Mismatch ─────────────────────────────────────

def check_category_mismatch(evidence: dict) -> list:
    """Flag words whose WordNet hypernyms don't pass through expected category."""
    conflicts = []

    for slot_key, slot_data in evidence["slots"].items():
        if slot_data.get("na"):
            continue

        for w in slot_data.get("words", []):
            key = f"{w['w']}::{w['pos']}"
            ev = slot_data.get("evidence", {}).get(key, {})

            if ev.get("category_compatible") is False:
                conflicts.append({
                    "type": "category_mismatch",
                    "key": key,
                    "atom": evidence["atom"],
                    "slot": slot_key,
                    "expected_category": evidence["category"],
                    "hypernym_chain": ev.get("hypernym_chain", []),
                    "severity": "MEDIUM",
                    "action": "Review: word may not semantically belong to this category.",
                })

    return conflicts


# ── B3c: Multi-Sense Ambiguity ─────────────────────────────────

def check_multi_sense_ambiguity(evidence: dict) -> list:
    """Flag words with high synset count (disambiguation risk)."""
    conflicts = []
    SYNSET_THRESHOLD = 5  # Words with 5+ synsets are ambiguous

    for slot_key, slot_data in evidence["slots"].items():
        if slot_data.get("na"):
            continue

        for w in slot_data.get("words", []):
            key = f"{w['w']}::{w['pos']}"
            ev = slot_data.get("evidence", {}).get(key, {})
            synset_count = len(ev.get("synsets", []))

            if synset_count >= SYNSET_THRESHOLD:
                conflicts.append({
                    "type": "multi_sense_ambiguity",
                    "key": key,
                    "atom": evidence["atom"],
                    "slot": slot_key,
                    "synset_count": synset_count,
                    "severity": "INFO",
                    "action": "Review: highly polysemous word. Ensure intended sense matches slot.",
                })

    return conflicts


# ── B3d: Intra-Atom Duplicate ──────────────────────────────────

def check_intra_atom_duplicate(evidence: dict) -> list:
    """Flag same lemma::pos appearing in multiple slots of one atom."""
    conflicts = []
    key_to_slots = {}

    for slot_key, slot_data in evidence["slots"].items():
        if slot_data.get("na"):
            continue
        for w in slot_data.get("words", []):
            key = f"{w['w']}::{w['pos']}"
            key_to_slots.setdefault(key, []).append(slot_key)

    for key, slots in key_to_slots.items():
        if len(slots) > 1:
            conflicts.append({
                "type": "intra_atom_duplicate",
                "key": key,
                "atom": evidence["atom"],
                "slots": slots,
                "severity": "INFO",
                "action": "May be valid (word spans multiple axis-levels). Flag for review.",
            })

    return conflicts


# ── B3e: WordNet Missing ───────────────────────────────────────

def check_wordnet_missing(evidence: dict) -> list:
    """Flag words not found in WordNet."""
    conflicts = []

    for slot_key, slot_data in evidence["slots"].items():
        if slot_data.get("na"):
            continue

        for w in slot_data.get("words", []):
            key = f"{w['w']}::{w['pos']}"
            ev = slot_data.get("evidence", {}).get(key, {})

            if not ev.get("wordnet_found", False):
                conflicts.append({
                    "type": "wordnet_missing",
                    "key": key,
                    "atom": evidence["atom"],
                    "slot": slot_key,
                    "severity": "LOW",
                    "action": "Word not in WordNet. May be valid (slang, neologism) or misspelled.",
                })

    return conflicts


# ── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Check conflicts in Lexicon evidence")
    parser.add_argument("--evidence", required=True, help="Evidence JSON for primary atom")
    parser.add_argument("--evidence-pair", help="Evidence JSON for symmetric pair atom")
    parser.add_argument("--atoms", required=True, help="Atoms definition JSON")
    parser.add_argument("--output", required=True, help="Output conflicts JSON")
    args = parser.parse_args()

    with open(args.evidence) as f:
        evidence = json.load(f)
    with open(args.atoms) as f:
        atoms = json.load(f)

    atom_id = evidence["atom"]
    print(f"Checking conflicts for {atom_id}")

    all_conflicts = []

    # B3a: Symmetric pair collision
    if args.evidence_pair:
        with open(args.evidence_pair) as f:
            evidence_pair = json.load(f)
        sym_conflicts = check_symmetric_collision(evidence, evidence_pair)
        all_conflicts.extend(sym_conflicts)
        print(f"  Symmetric collisions:   {len(sym_conflicts)}")
    else:
        print(f"  Symmetric collisions:   SKIPPED (no pair file)")

    # B3b: Category mismatch
    cat_conflicts = check_category_mismatch(evidence)
    all_conflicts.extend(cat_conflicts)
    print(f"  Category mismatches:    {len(cat_conflicts)}")

    # B3c: Multi-sense ambiguity
    sense_conflicts = check_multi_sense_ambiguity(evidence)
    all_conflicts.extend(sense_conflicts)
    print(f"  Multi-sense ambiguity:  {len(sense_conflicts)}")

    # B3d: Intra-atom duplicates
    dup_conflicts = check_intra_atom_duplicate(evidence)
    all_conflicts.extend(dup_conflicts)
    print(f"  Intra-atom duplicates:  {len(dup_conflicts)}")

    # B3e: WordNet missing
    missing_conflicts = check_wordnet_missing(evidence)
    all_conflicts.extend(missing_conflicts)
    print(f"  WordNet missing:        {len(missing_conflicts)}")

    # Summary
    severity_counts = {}
    for c in all_conflicts:
        sev = c["severity"]
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    print(f"\n  Total conflicts: {len(all_conflicts)}")
    for sev in ["HIGH", "MEDIUM", "INFO", "LOW"]:
        if sev in severity_counts:
            print(f"    {sev}: {severity_counts[sev]}")

    # Save
    output = {
        "atom": atom_id,
        "symmetric_pair": atoms["atoms"][atom_id]["symmetric_pair"],
        "total_conflicts": len(all_conflicts),
        "severity_summary": severity_counts,
        "conflicts": all_conflicts,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
