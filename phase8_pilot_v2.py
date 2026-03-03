#!/usr/bin/env python3
"""
ESDE Phase 8 — Molecule Generation Pilot v2 (Three-Phase Architecture)
=======================================================================

α: rule_generator.py   → Mechanical molecule generation (no LLM)
β: molecule_auditor.py  → Selective LLM audit (flagged molecules only)
γ: correction           → (future: re-gen or human review)

+ Path B: A1 Skeleton Builder (unchanged from v1)

Usage:
  cd esde/

  # Full run (rule gen + audit + skeleton)
  python3 phase8_pilot_v2.py --dataset mixed

  # Skip audit (rule gen + skeleton only, no LLM)
  python3 phase8_pilot_v2.py --dataset mixed --skip-audit

  # Limited run
  python3 phase8_pilot_v2.py --dataset mixed --max-articles 3

Output:
  output/phase8_pilot_v2/{dataset}/
    ├── molecules/          # Phase α draft molecules (JSONL per article)
    ├── audited/            # Phase β audit results (JSONL, flagged only)
    ├── skeletons/          # Path B: A1 skeletons (JSONL per article)
    ├── pilot_report.md
    └── pilot_stats.json
"""

import json
import math
import csv
import re
import sys
import os
import argparse
import time
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone


# ============================================================
# Import local modules
# ============================================================

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rule_generator import RuleGenerator, DraftMolecule, GenerationStats, run_batch, apply_dynamic_flags


# ============================================================
# 48 Slot IDs (for skeleton builder)
# ============================================================

SLOT_IDS = [
    "temporal.emergence", "temporal.establishment", "temporal.peak",
    "temporal.decline", "temporal.dissolution",
    "scale.individual", "scale.group", "scale.institution",
    "scale.civilization", "scale.universal",
    "ontological.material", "ontological.informational",
    "ontological.relational", "ontological.structural", "ontological.semantic",
    "interconnection.independent", "interconnection.catalytic",
    "interconnection.chained", "interconnection.synchronous", "interconnection.resonant",
    "resonance.superficial", "resonance.structural",
    "resonance.essential", "resonance.existential",
    "symmetry.destructive", "symmetry.inclusive",
    "symmetry.transformative", "symmetry.generative", "symmetry.cyclical",
    "lawfulness.causal", "lawfulness.emergent",
    "lawfulness.necessary", "lawfulness.contingent",
    "agency.reactive", "agency.adaptive",
    "agency.intentional", "agency.autonomous",
    "boundary.permeable", "boundary.selective",
    "boundary.rigid", "boundary.dissolved",
    "potential.latent", "potential.activated",
    "potential.kinetic", "potential.exhausted",
    "identity.generic", "identity.specific",
    "identity.archetypal",
]


# ============================================================
# Sentence Segmenter
# ============================================================

def segment_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    text = re.sub(r'^=+\s*.*?\s*=+$', '', text, flags=re.MULTILINE)
    pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    raw_sentences = re.split(pattern, text)
    
    sentences = []
    for s in raw_sentences:
        s = s.strip()
        if len(s) < 10:
            continue
        if not any(c.isalpha() for c in s):
            continue
        sentences.append(s)
    
    return sentences


# ============================================================
# A1 Skeleton Builder (unchanged from v1)
# ============================================================

class A1SkeletonBuilder:
    def __init__(self, centroids_path: str, a1_dir: str):
        self.centroids = self._load_centroids(centroids_path)
        self.word_vectors = self._load_a1_vectors(a1_dir)
        self.atom_ids = sorted(self.centroids.keys())
        print(f"  A1 Skeleton: {len(self.centroids)} centroids, "
              f"{len(self.word_vectors)} word vectors loaded")
    
    def _load_centroids(self, path: str) -> Dict[str, List[float]]:
        centroids = {}
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                centroids[row[0]] = [float(v) for v in row[1:]]
        return centroids
    
    def _load_a1_vectors(self, a1_dir: str) -> Dict[str, List[float]]:
        vectors = {}
        for f in sorted(Path(a1_dir).glob("*_a1_final.jsonl")):
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("status") == "Observation_Failed":
                        continue
                    rs = rec.get("raw_scores", {})
                    if not rs:
                        continue
                    word = rec.get("word", "").lower()
                    if word and word not in vectors:
                        vectors[word] = [rs.get(s, 0.0) for s in SLOT_IDS]
        return vectors
    
    def _cosine(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return dot / (na * nb)
    
    def build_skeleton(self, sentence: str, top_k: int = 3) -> Dict[str, Any]:
        tokens = re.sub(r"[^a-z0-9'\s]", " ", sentence.lower()).split()
        tokens = [t.strip("'") for t in tokens if t.strip("'") and len(t.strip("'")) >= 2]
        
        matched_tokens = []
        atom_scores = defaultdict(lambda: {"max_cos": 0.0, "source_word": "", "count": 0})
        
        for token in tokens:
            vec = self.word_vectors.get(token)
            if vec is None:
                continue
            matched_tokens.append(token)
            scores = [(aid, self._cosine(vec, self.centroids[aid])) for aid in self.atom_ids]
            scores.sort(key=lambda x: -x[1])
            for aid, cos_val in scores[:top_k]:
                if cos_val > atom_scores[aid]["max_cos"]:
                    atom_scores[aid]["max_cos"] = cos_val
                    atom_scores[aid]["source_word"] = token
                atom_scores[aid]["count"] += 1
        
        ranked = sorted(atom_scores.items(), key=lambda x: -x[1]["max_cos"])
        active_atoms = []
        seen = set()
        for aid, info in ranked:
            if aid in seen:
                continue
            seen.add(aid)
            active_atoms.append({
                "atom": aid, "cos_raw": round(info["max_cos"], 4),
                "source_word": info["source_word"], "token_hits": info["count"],
            })
            if len(active_atoms) >= 5:
                break
        
        coverage = len(matched_tokens) / len(tokens) if tokens else 0.0
        return {
            "sentence": sentence[:200],
            "tokens_total": len(tokens),
            "tokens_matched": len(matched_tokens),
            "coverage": round(coverage, 3),
            "active_atoms": active_atoms,
            "formula": None,
        }


# ============================================================
# Report Generator
# ============================================================

def generate_report(
    dataset_name: str,
    articles_processed: int,
    sentences_total: int,
    gen_stats: GenerationStats,
    audit_stats: Optional[Dict],
    skeleton_stats: Dict,
    out_dir: str,
):
    lines = []
    lines.append("# ESDE Phase 8 — Pilot Report v2 (Three-Phase Architecture)")
    lines.append("")
    lines.append(f"**Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**Dataset**: {dataset_name}")
    lines.append(f"**Articles**: {articles_processed}")
    lines.append(f"**Sentences**: {sentences_total}")
    lines.append("")
    
    # Phase α: Rule Generator
    lines.append("## Phase α: Rule-Based Generator (No LLM)")
    lines.append("")
    gs = gen_stats.to_dict()
    success_rate = gs["molecules_generated"] / max(gs["processed"], 1) * 100
    audit_rate = gs["audit_rate"] * 100
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Processed | {gs['processed']} |")
    lines.append(f"| Molecules generated | {gs['molecules_generated']} |")
    lines.append(f"| **Success rate** | **{success_rate:.1f}%** |")
    lines.append(f"| No candidates | {gs['no_candidates']} |")
    lines.append(f"| SVO triples extracted | {gs['svo_extracted']} |")
    lines.append(f"| SVO grounded to atoms | {gs['svo_grounded']} |")
    lines.append(f"| Flagged for audit | {gs['flagged_for_audit']} |")
    lines.append(f"| **Audit rate** | **{audit_rate:.1f}%** |")
    lines.append("")
    
    # Formula patterns
    if gs["formula_patterns"]:
        lines.append("### Formula Patterns")
        lines.append("")
        lines.append("| Pattern | Count |")
        lines.append("|---------|-------|")
        for pat, cnt in sorted(gs["formula_patterns"].items(), key=lambda x: -x[1])[:20]:
            lines.append(f"| `{pat}` | {cnt} |")
        lines.append("")
    
    # Flag distribution
    if gs["flag_counts"]:
        lines.append("### Audit Flag Distribution")
        lines.append("")
        lines.append("| Flag | Count |")
        lines.append("|------|-------|")
        for flag, cnt in sorted(gs["flag_counts"].items(), key=lambda x: -x[1]):
            lines.append(f"| {flag} | {cnt} |")
        lines.append("")
    
    # Phase β: Audit
    if audit_stats:
        lines.append("## Phase β: LLM Audit (Selective)")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in audit_stats.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")
    else:
        lines.append("## Phase β: LLM Audit")
        lines.append("")
        lines.append("*Skipped (--skip-audit)*")
        lines.append("")
    
    # Path B: Skeleton
    lines.append("## Path B: A1 Skeleton Builder (No LLM)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Processed | {skeleton_stats['processed']} |")
    lines.append(f"| Avg A1 coverage | {skeleton_stats.get('avg_coverage', 0):.1%} |")
    lines.append(f"| Avg atoms/skeleton | {skeleton_stats.get('avg_atoms', 0):.1f} |")
    lines.append("")
    
    # Top atoms
    if skeleton_stats.get("atom_frequency"):
        lines.append("### Top 20 Atoms (skeleton)")
        lines.append("")
        lines.append("| Atom | Count |")
        lines.append("|------|-------|")
        for atom, count in skeleton_stats["atom_frequency"].most_common(20):
            lines.append(f"| {atom} | {count} |")
        lines.append("")
    
    lines.append("## Observations & Next Steps")
    lines.append("")
    lines.append("*(To be filled after reviewing output files)*")
    
    with open(os.path.join(out_dir, "pilot_report.md"), "w") as f:
        f.write("\n".join(lines))
    
    # Stats JSON
    stats_json = {
        "dataset": dataset_name,
        "articles": articles_processed,
        "sentences": sentences_total,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase_alpha": gs,
        "phase_beta": audit_stats,
        "path_b": {
            k: v for k, v in skeleton_stats.items()
            if k not in ("atom_frequency", "category_dist")
        },
    }
    if skeleton_stats.get("atom_frequency"):
        stats_json["path_b"]["top_atoms"] = dict(skeleton_stats["atom_frequency"].most_common(50))
    
    with open(os.path.join(out_dir, "pilot_stats.json"), "w") as f:
        json.dump(stats_json, f, indent=2, ensure_ascii=False)
    
    print(f"  ✅ Report + stats written to {out_dir}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="ESDE Phase 8 — Pilot v2 (Three-Phase Architecture)"
    )
    parser.add_argument("--dataset", default="mixed")
    parser.add_argument("--max-articles", type=int, default=0)
    parser.add_argument("--max-sentences", type=int, default=0)
    parser.add_argument("--skip-audit", action="store_true",
                        help="Skip Phase β (no LLM needed)")
    parser.add_argument("--a1-centroids",
                        default="integration/lexicon/synapse_v4_report/atom_centroids_48d_raw.csv")
    parser.add_argument("--a1-dir", default="integration/lexicon/audit_output/")
    parser.add_argument("--synapse", default="esde_synapses_v3.json")
    parser.add_argument("--glossary", default="glossary_results.json")
    parser.add_argument("--out-dir", default="output/phase8_pilot_v2")
    args = parser.parse_args()
    
    print("=" * 70)
    print("  ESDE Phase 8 — Pilot v2 (Three-Phase Architecture)")
    print("  α: Rule Generator | β: LLM Audit | B: A1 Skeleton")
    print("=" * 70)
    
    # ── Load data ──
    print(f"\n[1/5] Loading dataset '{args.dataset}'...")
    from harvester.storage import load_dataset
    articles = load_dataset(args.dataset)
    
    article_ids = sorted(articles.keys())
    if args.max_articles > 0:
        article_ids = article_ids[:args.max_articles]
    
    print(f"  {len(articles)} available, {len(article_ids)} to process")
    
    # ── Init Rule Generator ──
    print(f"\n[2/5] Initializing Rule Generator (Phase α)...")
    rule_gen = RuleGenerator(
        synapse_file=args.synapse,
        glossary_file=args.glossary,
    )
    print(f"  ✅ Rule Generator ready")
    
    # ── Init Auditor ──
    auditor = None
    if not args.skip_audit:
        print(f"\n[3/5] Initializing Molecule Auditor (Phase β)...")
        try:
            from molecule_auditor import MoleculeAuditor
            auditor = MoleculeAuditor()
            # Quick check LLM
            import urllib.request
            req = urllib.request.Request(f"{auditor.llm_host}/models", method="GET")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    print(f"  ✅ Auditor ready (LLM available)")
                else:
                    print(f"  ⚠ LLM not available, skipping audit")
                    auditor = None
        except Exception as e:
            print(f"  ⚠ Auditor init failed: {e}")
            auditor = None
    else:
        print(f"\n[3/5] Auditor skipped (--skip-audit)")
    
    # ── Init Skeleton Builder ──
    print(f"\n[4/5] Initializing A1 Skeleton Builder...")
    skeleton_builder = A1SkeletonBuilder(
        centroids_path=args.a1_centroids,
        a1_dir=args.a1_dir,
    )
    
    # ── Process ──
    print(f"\n[5/5] Processing articles...")
    
    out_base = os.path.join(args.out_dir, args.dataset)
    mol_dir = os.path.join(out_base, "molecules")
    audit_dir = os.path.join(out_base, "audited")
    skel_dir = os.path.join(out_base, "skeletons")
    os.makedirs(mol_dir, exist_ok=True)
    os.makedirs(audit_dir, exist_ok=True)
    os.makedirs(skel_dir, exist_ok=True)
    
    total_sentences = 0
    skel_stats = {
        "processed": 0, "atom_frequency": Counter(),
        "category_dist": Counter(),
    }
    _cov_sum = 0.0
    _atoms_sum = 0
    all_threshold_reports = []
    
    for article_idx, article_id in enumerate(article_ids, 1):
        text = articles[article_id]
        sentences = segment_sentences(text)
        if args.max_sentences > 0:
            sentences = sentences[:args.max_sentences]
        total_sentences += len(sentences)
        
        print(f"\n  [{article_idx}/{len(article_ids)}] {article_id}: {len(sentences)} sentences")
        
        # ── Phase α: Rule Generator (2-pass) ──
        t0 = time.time()
        mol_path = os.path.join(mol_dir, f"{article_id}_molecules.jsonl")
        audit_path = os.path.join(audit_dir, f"{article_id}_audited.jsonl")
        
        # Pass 1 + Pass 2 via run_batch
        molecules, _, threshold_report = run_batch(sentences, rule_gen, article_id)
        all_threshold_reports.append({
            "article_id": article_id,
            "report": threshold_report,
        })
        
        # Write molecules + collect flagged for batch audit
        flagged_mols = []
        with open(mol_path, "w") as mf:
            for d in molecules:
                mf.write(json.dumps(d, ensure_ascii=False) + "\n")
                if d.get("needs_audit", False):
                    flagged_mols.append(d)
        
        # Phase β: Batch audit flagged molecules (8 parallel)
        if auditor and flagged_mols:
            print(f"    Phase β: auditing {len(flagged_mols)} flagged molecules (8 parallel)...")
            audit_results = auditor.batch_audit(flagged_mols, max_workers=8)
            
            with open(audit_path, "w") as af:
                for mol_d, verdict in audit_results:
                    audit_entry = {
                        "article_id": article_id,
                        "sentence_idx": mol_d.get("sentence_idx", -1),
                        "sentence": mol_d.get("source_text", "")[:200],
                        "flags": mol_d.get("meta", {}).get("flags", []),
                        "verdict": verdict.to_dict(),
                    }
                    af.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")
        
        elapsed_a = time.time() - t0
        flagged = sum(1 for d in molecules if d.get("needs_audit", False))
        audit_pct = flagged / max(len(molecules), 1) * 100
        print(f"    Phase α: {elapsed_a:.1f}s | {len(molecules)} molecules, "
              f"{flagged} flagged ({audit_pct:.0f}%)")
        
        # ── Path B: Skeleton ──
        t0 = time.time()
        skel_path = os.path.join(skel_dir, f"{article_id}_skeletons.jsonl")
        with open(skel_path, "w") as sf:
            for sent_idx, sentence in enumerate(sentences):
                skeleton = skeleton_builder.build_skeleton(sentence)
                skeleton["article_id"] = article_id
                skeleton["sentence_idx"] = sent_idx
                sf.write(json.dumps(skeleton, ensure_ascii=False) + "\n")
                
                skel_stats["processed"] += 1
                _cov_sum += skeleton["coverage"]
                _atoms_sum += len(skeleton["active_atoms"])
                
                for aa in skeleton["active_atoms"]:
                    atom = aa["atom"]
                    skel_stats["atom_frequency"][atom] += 1
                    cat = atom.split(".")[0] if "." in atom else "?"
                    skel_stats["category_dist"][cat] += 1
        
        elapsed_b = time.time() - t0
        print(f"    Path B:  {elapsed_b:.1f}s")
    
    # Compute skeleton averages
    n = skel_stats["processed"]
    if n > 0:
        skel_stats["avg_coverage"] = _cov_sum / n
        skel_stats["avg_atoms"] = _atoms_sum / n
    
    # ── Report ──
    print("\n" + "=" * 70)
    print("  Generating Report")
    print("=" * 70)
    
    generate_report(
        dataset_name=args.dataset,
        articles_processed=len(article_ids),
        sentences_total=total_sentences,
        gen_stats=rule_gen.stats,
        audit_stats=auditor.stats if auditor else None,
        skeleton_stats=skel_stats,
        out_dir=out_base,
    )
    
    # Save threshold reports
    thresh_path = os.path.join(out_base, "threshold_reports.json")
    with open(thresh_path, "w") as f:
        json.dump(all_threshold_reports, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Threshold reports: {thresh_path}")
    
    # ── Summary ──
    gs = rule_gen.stats
    print("\n" + "=" * 70)
    print("  PILOT v2 COMPLETE")
    print("=" * 70)
    print(f"  Articles: {len(article_ids)}")
    print(f"  Sentences: {total_sentences}")
    print(f"  Phase α: {gs.molecules_generated} molecules "
          f"({gs.molecules_generated / max(gs.processed, 1) * 100:.1f}% success), "
          f"{gs.flagged_for_audit} flagged ({gs.flagged_for_audit / max(gs.molecules_generated, 1) * 100:.1f}%)")
    if auditor:
        print(f"  Phase β: {auditor.stats['llm_called']} LLM calls, "
              f"{auditor.stats['llm_pass']} pass, {auditor.stats['llm_fail']} fail")
    print(f"  Path B: avg coverage {skel_stats.get('avg_coverage', 0):.1%}")
    print(f"\n  Output: {out_base}/")


if __name__ == "__main__":
    main()