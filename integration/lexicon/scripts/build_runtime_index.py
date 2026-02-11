#!/usr/bin/env python3
"""
Phase C Step C4: Build Runtime Index from Lexicon Master
Generates the reverse lookup index used by LexiconGrounder.

CRITICAL (Final Patch B): Only status=="core" words enter the index.
CRITICAL (Final Patch C): Normalization applied at generation time (second defense).

Usage:
    python build_runtime_index.py \
        --master ../output/lexicon_master.json \
        --output ../output/lexicon_index.json
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def normalize_word(w: str) -> str:
    """Final Patch C: normalize at index generation (second defense)."""
    w = w.strip().lower()
    w = w.replace("_", " ")
    w = re.sub(r'\s+', ' ', w)
    return w


def build_index(master: dict) -> dict:
    """
    Build reverse lookup: lemma::pos → [{atom, axis, level}, ...]
    Only includes status=="core" words (Final Patch B).
    """
    index = {}
    stats = {
        "total_core": 0,
        "total_proposed": 0,
        "total_rejected": 0,
        "atoms_processed": 0,
        "active_slots": 0,
        "na_slots": 0,
    }

    for atom_id, atom_data in master.items():
        if atom_id == "meta":
            continue

        stats["atoms_processed"] += 1

        for slot_key, slot_data in atom_data.get("slots", {}).items():
            if slot_data.get("na", False):
                stats["na_slots"] += 1
                continue

            stats["active_slots"] += 1
            parts = slot_key.split(".", 1)
            if len(parts) != 2:
                continue
            axis, level = parts

            for word_entry in slot_data.get("words", []):
                status = word_entry.get("status", "proposed")

                if status == "core":
                    stats["total_core"] += 1
                    w = normalize_word(word_entry["w"])
                    pos = word_entry["pos"]
                    key = f"{w}::{pos}"

                    if key not in index:
                        index[key] = []

                    # Avoid exact duplicate entries
                    entry = {"atom": atom_id, "axis": axis, "level": level}
                    if entry not in index[key]:
                        index[key].append(entry)

                elif status == "proposed":
                    stats["total_proposed"] += 1
                elif status == "rejected":
                    stats["total_rejected"] += 1

    return index, stats


def main():
    parser = argparse.ArgumentParser(description="Build Runtime Index from Lexicon Master")
    parser.add_argument("--master", required=True, help="Lexicon Master JSON")
    parser.add_argument("--output", required=True, help="Output index JSON")
    args = parser.parse_args()

    with open(args.master) as f:
        master = json.load(f)

    print("Building runtime index...")
    index, stats = build_index(master)

    print(f"\nStats:")
    print(f"  Atoms processed: {stats['atoms_processed']}")
    print(f"  Active slots:    {stats['active_slots']}")
    print(f"  N/A slots:       {stats['na_slots']}")
    print(f"  Core words:      {stats['total_core']} → indexed")
    print(f"  Proposed words:  {stats['total_proposed']} → NOT indexed")
    print(f"  Rejected words:  {stats['total_rejected']} → NOT indexed")
    print(f"  Unique keys:     {len(index)}")

    # Multi-placement stats
    multi = sum(1 for v in index.values() if len(v) > 1)
    print(f"  Multi-placement: {multi} keys appear in 2+ locations")

    # Save
    output = {
        "version": "1.0",
        "generated_from": str(args.master),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stats": stats,
        "index": index,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
