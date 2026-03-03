#!/usr/bin/env python3
"""
ESDE Lexicon v2 — Batch Report Generator
==========================================
Run after run_all_atoms.sh completes.
Generates statistics across all atoms.

Usage:
  python3 batch_report.py
  python3 batch_report.py --audit-dir audit_output --out batch_report.md
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_final_records(audit_dir: str):
    """Load all *_a1_final.jsonl files."""
    records_by_atom = {}
    for f in sorted(Path(audit_dir).glob("*_a1_final.jsonl")):
        atom_key = f.stem.replace("_a1_final", "").replace("_", ".")
        records = []
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if records:
            # Get actual atom ID from first record
            actual_atom = records[0].get("atom", atom_key)
            records_by_atom[actual_atom] = records
    return records_by_atom


def load_audit_records(audit_dir: str):
    """Load all *_audit.jsonl files."""
    audits_by_atom = {}
    for f in sorted(Path(audit_dir).glob("*_audit.jsonl")):
        atom_key = f.stem.replace("_audit", "").replace("_", ".")
        records = []
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if records:
            actual_atom = records[0].get("atom", atom_key)
            audits_by_atom[actual_atom] = records
    return audits_by_atom


def analyze(records_by_atom, audits_by_atom):
    """Compute cross-atom statistics."""
    stats = {
        "total_atoms": len(records_by_atom),
        "total_words": 0,
        "total_reobserved": 0,
        "total_diffuse": 0,
        "total_mass_fail": 0,
        "total_pass_diffuse": 0,
        "atoms": [],
        "flag_freq": defaultdict(int),
        "c4c5_by_category": defaultdict(lambda: defaultdict(int)),
        "c4c5_words_by_category": defaultdict(int),
        "rare_words": [],  # GPT-4: agnize-type candidates
        "mass_fail_reasons": defaultdict(int),
        "category_stats": defaultdict(lambda: {
            "atoms": 0, "words": 0, "revise_rate": [],
            "avg_focus": [], "diffuse": 0, "mass_fail": 0,
        }),
    }

    all_focus = []
    all_sums = []
    all_nz = []

    for atom_id in sorted(records_by_atom.keys()):
        records = records_by_atom[atom_id]
        audits = audits_by_atom.get(atom_id, [])
        category = atom_id.split(".")[0]

        # Filter out records without raw_scores (Observation_Failed etc.)
        valid_records = [r for r in records if "raw_scores" in r]
        failed_records = [r for r in records if "raw_scores" not in r]

        n_words = len(valid_records)
        if n_words == 0:
            continue

        reobs = [r for r in valid_records if r.get("re_observed")]
        n_reobs = len(reobs)
        n_revise = sum(1 for a in audits if a.get("final_status") == "REVISE")

        sums = [sum(r["raw_scores"].values()) for r in valid_records]
        nzs = [sum(1 for v in r["raw_scores"].values() if v > 0) for r in valid_records]
        fs = [r.get("focus_rate", 0) for r in valid_records]
        n_diffuse = sum(1 for r in valid_records if r.get("status") == "Diffuse_Observation")

        # Mass guard fails + reason codes + rare words
        n_mass_fail = 0
        for r in reobs:
            trace = r.get("re_observe_trace", {})
            mass = trace.get("min_mass_guard", {})
            nz = trace.get("final_nonzero", 0)
            if not mass.get("pass", True):
                n_mass_fail += 1
                reason = mass.get("reason", "unknown")
                stats["mass_fail_reasons"][reason] += 1
            # Rare word detection: mass fail OR very thin (nz <= 2)
            if not mass.get("pass", True) or nz <= 2:
                stats["rare_words"].append({
                    "word": r["word"],
                    "pos": r.get("pos", "?"),
                    "atom": atom_id,
                    "final_nz": nz,
                    "final_sum": trace.get("final_sum", 0),
                    "mass_pass": mass.get("pass", True),
                    "reason": mass.get("reason"),
                })

        # PASS but Diffuse count
        n_pass_diffuse = 0
        for r, a in zip(records, audits) if len(audits) == len(records) else []:
            pass
        # Simpler: count from final records
        for r in valid_records:
            if not r.get("re_observed") and r.get("status") == "Diffuse_Observation":
                n_pass_diffuse += 1
        stats["total_pass_diffuse"] += n_pass_diffuse

        # Flag frequency + C4/C5 by category
        for a in audits:
            for f in a.get("pre_screen_flags", []):
                stats["flag_freq"][f] += 1
                if f.startswith("C4_") or f.startswith("C5_"):
                    stats["c4c5_by_category"][category][f] += 1
        stats["c4c5_words_by_category"][category] += n_words

        revise_rate = n_revise / n_words if n_words > 0 else 0

        atom_stat = {
            "atom": atom_id,
            "category": category,
            "words": n_words,
            "revise": n_revise,
            "revise_rate": revise_rate,
            "reobserved": n_reobs,
            "diffuse": n_diffuse,
            "mass_fail": n_mass_fail,
            "avg_sum": sum(sums) / len(sums) if sums else 0,
            "avg_nz": sum(nzs) / len(nzs) if nzs else 0,
            "avg_focus": sum(fs) / len(fs) if fs else 0,
            "min_focus": min(fs) if fs else 0,
            "max_focus": max(fs) if fs else 0,
        }
        stats["atoms"].append(atom_stat)

        stats["total_words"] += n_words
        stats["total_reobserved"] += n_reobs
        stats["total_diffuse"] += n_diffuse
        stats["total_mass_fail"] += n_mass_fail

        all_focus.extend(fs)
        all_sums.extend(sums)
        all_nz.extend(nzs)

        # Category stats
        cs = stats["category_stats"][category]
        cs["atoms"] += 1
        cs["words"] += n_words
        cs["revise_rate"].append(revise_rate)
        cs["avg_focus"].extend(fs)
        cs["diffuse"] += n_diffuse
        cs["mass_fail"] += n_mass_fail

    stats["overall_avg_focus"] = sum(all_focus) / len(all_focus) if all_focus else 0
    stats["overall_avg_sum"] = sum(all_sums) / len(all_sums) if all_sums else 0
    stats["overall_avg_nz"] = sum(all_nz) / len(all_nz) if all_nz else 0

    return stats


def format_report(stats):
    """Format as Markdown report."""
    lines = []
    lines.append("# ESDE Lexicon v2 — Batch Report")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- Atoms processed: **{stats['total_atoms']}** / 326")
    lines.append(f"- Total words: **{stats['total_words']}**")
    lines.append(f"- Re-observed (3-condition): **{stats['total_reobserved']}**")
    lines.append(f"- Diffuse observations: **{stats['total_diffuse']}**")
    lines.append(f"- PASS but Diffuse (not re-observed): **{stats['total_pass_diffuse']}**")
    lines.append(f"- Min mass guard fails: **{stats['total_mass_fail']}**")
    lines.append(f"- Rare/thin words: **{len(stats['rare_words'])}**")
    lines.append(f"- Avg Focus Rate: **{stats['overall_avg_focus']:.4f}**")
    lines.append(f"- Avg Sum: **{stats['overall_avg_sum']:.1f}**")
    lines.append(f"- Avg Nonzero: **{stats['overall_avg_nz']:.1f}**")
    lines.append("")

    # Category summary
    lines.append("## By Category")
    lines.append("")
    lines.append("| Category | Atoms | Words | Avg Revise% | Avg Focus | Diffuse | Mass Fail |")
    lines.append("|----------|-------|-------|-------------|-----------|---------|-----------|")
    for cat in sorted(stats["category_stats"].keys()):
        cs = stats["category_stats"][cat]
        avg_rr = sum(cs["revise_rate"]) / len(cs["revise_rate"]) * 100 if cs["revise_rate"] else 0
        avg_f = sum(cs["avg_focus"]) / len(cs["avg_focus"]) if cs["avg_focus"] else 0
        lines.append(f"| {cat:8s} | {cs['atoms']:5d} | {cs['words']:5d} | {avg_rr:10.1f}% | {avg_f:9.4f} | {cs['diffuse']:7d} | {cs['mass_fail']:9d} |")
    lines.append("")

    # [GPT-3] C4/C5 by category
    if stats["c4c5_by_category"]:
        lines.append("## C4/C5 Flag Rates by Category")
        lines.append("")
        lines.append("Semantic terrain map: shows which categories naturally trigger axis-generic (C4) or POS-coherence (C5) flags.")
        lines.append("")
        lines.append("| Category | Words | Flag | Count | Rate |")
        lines.append("|----------|-------|------|-------|------|")
        for cat in sorted(stats["c4c5_by_category"].keys()):
            total_w = stats["c4c5_words_by_category"].get(cat, 1)
            for flag, count in sorted(stats["c4c5_by_category"][cat].items(), key=lambda x: -x[1]):
                rate = count / total_w * 100
                lines.append(f"| {cat:8s} | {total_w:5d} | `{flag}` | {count:5d} | {rate:5.1f}% |")
        lines.append("")

    # Flag frequency (all)
    lines.append("## Flag Frequencies (all atoms)")
    lines.append("")
    for f, c in sorted(stats["flag_freq"].items(), key=lambda x: -x[1]):
        lines.append(f"- `{f}`: {c}")
    lines.append("")

    # Mass fail reason breakdown
    if stats["mass_fail_reasons"]:
        lines.append("## Mass Guard Fail Reasons")
        lines.append("")
        for reason, count in sorted(stats["mass_fail_reasons"].items(), key=lambda x: -x[1]):
            lines.append(f"- `{reason}`: {count}")
        lines.append("")

    # Worst atoms (highest revise rate)
    lines.append("## Atoms with Highest REVISE Rate (top 20)")
    lines.append("")
    lines.append("| Atom | Words | REVISE% | Avg Focus | Diffuse | Mass Fail |")
    lines.append("|------|-------|---------|-----------|---------|-----------|")
    by_revise = sorted(stats["atoms"], key=lambda x: -x["revise_rate"])
    for a in by_revise[:20]:
        lines.append(f"| {a['atom']:20s} | {a['words']:5d} | {a['revise_rate']*100:6.1f}% | {a['avg_focus']:9.4f} | {a['diffuse']:7d} | {a['mass_fail']:9d} |")
    lines.append("")

    # Mass fail atoms
    mass_fails = [a for a in stats["atoms"] if a["mass_fail"] > 0]
    if mass_fails:
        lines.append("## Atoms with Mass Guard Fails")
        lines.append("")
        for a in mass_fails:
            lines.append(f"- **{a['atom']}**: {a['mass_fail']} fail(s) out of {a['reobserved']} re-observed")
        lines.append("")

    # Diffuse atoms
    diffuse_atoms = [a for a in stats["atoms"] if a["diffuse"] > 0]
    if diffuse_atoms:
        lines.append("## Atoms with Diffuse Observations")
        lines.append("")
        for a in sorted(diffuse_atoms, key=lambda x: -x["diffuse"]):
            lines.append(f"- **{a['atom']}**: {a['diffuse']} diffuse out of {a['words']} words (avg F={a['avg_focus']:.4f})")
        lines.append("")

    # [GPT-4] Rare/thin word list (Seed improvement candidates)
    if stats["rare_words"]:
        lines.append("## Rare/Thin Words (Seed Improvement Candidates)")
        lines.append("")
        lines.append("Words where 3-condition intersection left insufficient mass.")
        lines.append("These indicate vocabulary supply gaps or words too ambiguous for current atom assignment.")
        lines.append("")
        lines.append("| Word | POS | Atom | NZ | Sum | Mass | Reason |")
        lines.append("|------|-----|------|----|-----|------|--------|")
        for rw in sorted(stats["rare_words"], key=lambda x: x["final_sum"]):
            mass_str = "✅" if rw["mass_pass"] else "❌"
            reason = rw.get("reason") or "thin_nz"
            lines.append(f"| {rw['word']:20s} | {rw['pos']:4s} | {rw['atom']:15s} | {rw['final_nz']:2d} | {rw['final_sum']:3.0f} | {mass_str} | `{reason}` |")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="ESDE Batch Report Generator")
    parser.add_argument("--audit-dir", default="audit_output", help="Directory with audit/final JSONL files")
    parser.add_argument("--out", default="batch_report.md", help="Output report file")
    args = parser.parse_args()

    print(f"Loading results from {args.audit_dir}/...")
    records = load_final_records(args.audit_dir)
    audits = load_audit_records(args.audit_dir)

    if not records:
        print("ERROR: No final JSONL files found")
        return

    print(f"  Found {len(records)} atoms")
    stats = analyze(records, audits)
    report = format_report(stats)

    with open(args.out, "w") as f:
        f.write(report)

    print(f"\nReport written to {args.out}")

    # Print summary to console too
    print(f"\n{'='*50}")
    print(f"  Atoms: {stats['total_atoms']}")
    print(f"  Words: {stats['total_words']}")
    print(f"  Re-observed: {stats['total_reobserved']}")
    print(f"  Diffuse: {stats['total_diffuse']}")
    print(f"  Mass fails: {stats['total_mass_fail']}")
    print(f"  Avg Focus: {stats['overall_avg_focus']:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()