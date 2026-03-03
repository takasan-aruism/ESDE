#!/usr/bin/env python3
"""
ESDE Phase 8 — Molecule Generation Pilot (Option C)
=====================================================
Connects Harvester cached data to Phase 8 pipeline.

Two paths run in parallel on the same segments:
  Path A: Current pipeline (Sensor v2 → Generator LLM → Validator)
  Path B: A1 Skeleton Builder (centroid lookup, no LLM)

Usage:
  cd esde/

  # Both paths (requires QwQ running)
  python3 phase8_pilot.py --dataset mixed --max-articles 3

  # Skeleton only (no LLM needed)
  python3 phase8_pilot.py --dataset mixed --max-articles 3 --skeleton-only

  # Full dataset
  python3 phase8_pilot.py --dataset mixed

Output:
  output/phase8_pilot/{dataset}/
    ├── molecules/          # Path A: LLM-generated molecules (JSONL per article)
    ├── skeletons/          # Path B: A1 skeleton output (JSONL per article)
    ├── ledger.jsonl        # Path A: Ledger entries
    ├── pilot_report.md     # Comparison report
    └── pilot_stats.json    # Machine-readable statistics
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
# 48 Slot IDs (canonical order)
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
# Sentence Segmenter (lightweight, no spaCy dependency)
# ============================================================

def segment_sentences(text: str) -> List[str]:
    """
    Split text into sentences. Simple rule-based approach.
    Handles common abbreviations and decimal numbers.
    """
    # Remove Wikipedia section headers (== Title ==)
    text = re.sub(r'^=+\s*.*?\s*=+$', '', text, flags=re.MULTILINE)
    
    # Split on sentence boundaries
    # Negative lookbehind for common abbreviations
    pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    raw_sentences = re.split(pattern, text)
    
    sentences = []
    for s in raw_sentences:
        s = s.strip()
        if len(s) < 10:  # Skip very short fragments
            continue
        if not any(c.isalpha() for c in s):  # Skip non-text
            continue
        sentences.append(s)
    
    return sentences


# ============================================================
# A1 Skeleton Builder (Path B — No LLM)
# ============================================================

class A1SkeletonBuilder:
    """
    Build molecule skeletons using A1 48D centroid lookup.
    No LLM required. Fast dictionary-based lookup.
    """
    
    def __init__(self, centroids_path: str, a1_dir: str):
        """
        Args:
            centroids_path: atom_centroids_48d_raw.csv
            a1_dir: Directory with *_a1_final.jsonl files
        """
        self.centroids = self._load_centroids(centroids_path)
        self.word_vectors = self._load_a1_vectors(a1_dir)
        self.atom_ids = sorted(self.centroids.keys())
        
        print(f"  A1 Skeleton: {len(self.centroids)} centroids, "
              f"{len(self.word_vectors)} word vectors loaded")
    
    def _load_centroids(self, path: str) -> Dict[str, List[float]]:
        centroids = {}
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)  # header
            for row in reader:
                centroids[row[0]] = [float(v) for v in row[1:]]
        return centroids
    
    def _load_a1_vectors(self, a1_dir: str) -> Dict[str, List[float]]:
        """Load word → 48D vector from A1 final data. Keyed by lowercase word."""
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
                        # Use first occurrence (highest quality)
                        vectors[word] = [rs.get(s, 0.0) for s in SLOT_IDS]
        return vectors
    
    def _cosine(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return dot / (na * nb)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer matching Sensor V2 approach."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9'\s]", " ", text)
        tokens = text.split()
        return [t.strip("'") for t in tokens if t.strip("'") and len(t.strip("'")) >= 2]
    
    def build_skeleton(self, sentence: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Build skeleton from sentence using A1 48D lookup.
        
        Returns:
            {
                "sentence": str,
                "tokens_total": int,
                "tokens_matched": int,
                "coverage": float,
                "active_atoms": [{atom, cos_raw, source_word, rank}, ...],
                "formula": null (skeleton only)
            }
        """
        tokens = self._tokenize(sentence)
        
        # Look up each token in A1 vocabulary
        matched_tokens = []
        atom_scores = defaultdict(lambda: {"max_cos": 0.0, "source_word": "", "count": 0})
        
        for token in tokens:
            vec = self.word_vectors.get(token)
            if vec is None:
                continue
            
            matched_tokens.append(token)
            
            # Find top-k atoms for this token
            scores = []
            for aid in self.atom_ids:
                c = self._cosine(vec, self.centroids[aid])
                scores.append((aid, c))
            scores.sort(key=lambda x: -x[1])
            
            for aid, cos_val in scores[:top_k]:
                if cos_val > atom_scores[aid]["max_cos"]:
                    atom_scores[aid]["max_cos"] = cos_val
                    atom_scores[aid]["source_word"] = token
                atom_scores[aid]["count"] += 1
        
        # Select top atoms (by max cosine across all tokens)
        ranked_atoms = sorted(atom_scores.items(), key=lambda x: -x[1]["max_cos"])
        
        # Deduplicate: keep top 5 unique atoms
        active_atoms = []
        seen = set()
        for aid, info in ranked_atoms:
            if aid in seen:
                continue
            seen.add(aid)
            active_atoms.append({
                "atom": aid,
                "cos_raw": round(info["max_cos"], 4),
                "source_word": info["source_word"],
                "token_hits": info["count"],
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
            "formula": None,  # Skeleton only — no operator assignment
        }


# ============================================================
# Path A: Current Pipeline Wrapper
# ============================================================

def try_import_pipeline():
    """
    Try to import existing Phase 8 pipeline components.
    Returns (sensor, generator, validator) or None if unavailable.
    """
    try:
        from sensor.esde_sensor_v2_modular import ESDESensorV2
        from sensor.molecule_generator_live import MoleculeGeneratorLive
        from sensor.validator_v83 import MoleculeValidatorV83
        return {
            "SensorClass": ESDESensorV2,
            "GeneratorClass": MoleculeGeneratorLive,
            "ValidatorClass": MoleculeValidatorV83,
        }
    except ImportError as e:
        print(f"  ⚠ Pipeline import failed: {e}")
        return None


def check_llm_available(host: str = "http://100.107.6.119:8001/v1") -> bool:
    """Check if QwQ LLM server is reachable."""
    try:
        import urllib.request
        req = urllib.request.Request(f"{host}/models", method="GET")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


class PipelineRunner:
    """Wraps existing Phase 8 pipeline for batch execution."""
    
    def __init__(self, synapse_file: str, glossary_file: str,
                 synapse_patches: List[str] = None):
        components = try_import_pipeline()
        if components is None:
            raise RuntimeError("Pipeline components not importable")
        
        self.sensor = components["SensorClass"](
            synapse_file=synapse_file,
            glossary_file=glossary_file,
        )
                
        # Load glossary for generator
        with open(glossary_file) as f:
            glossary = json.load(f)
        if "glossary" in glossary:
            glossary = glossary["glossary"]
                
        self.generator = components["GeneratorClass"](glossary=glossary)
        self.validator = components["ValidatorClass"] if "ValidatorClass" in components else None
        
        self.stats = {"processed": 0, "success": 0, "failed": 0, "abstained": 0}
    
    def process_sentence(self, sentence: str) -> Optional[Dict]:
        """Run single sentence through pipeline. Returns molecule or None."""
        self.stats["processed"] += 1
        
        try:
            # Sensor
            sensor_result = self.sensor.analyze(sentence)
            candidates = sensor_result.get("candidates", [])
            
            if not candidates:
                self.stats["abstained"] += 1
                return None
            
            # Generator (LLM call)
            gen_result = self.generator.generate(
                original_text=sentence,
                candidates=candidates,
            )
            
            if not gen_result.success:
                self.stats["failed"] += 1
                return {
                    "sentence": sentence[:200],
                    "status": "generation_failed",
                    "error": gen_result.error,
                    "candidates_count": len(candidates),
                }
            
            self.stats["success"] += 1
            
            molecule = gen_result.molecule
            molecule["_meta_pilot"] = {
                "sentence": sentence[:200],
                "candidates_count": len(candidates),
                "status": "success",
            }
            
            return molecule
            
        except Exception as e:
            self.stats["failed"] += 1
            return {
                "sentence": sentence[:200],
                "status": "error",
                "error": str(e),
            }


# ============================================================
# Report Generator
# ============================================================

def generate_report(
    dataset_name: str,
    articles_processed: int,
    sentences_total: int,
    path_a_stats: Optional[Dict],
    path_b_stats: Dict,
    out_dir: str,
):
    """Generate pilot_report.md and pilot_stats.json."""
    
    lines = []
    lines.append("# ESDE Phase 8 — Molecule Generation Pilot Report")
    lines.append("")
    lines.append(f"**Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**Dataset**: {dataset_name}")
    lines.append(f"**Articles**: {articles_processed}")
    lines.append(f"**Sentences**: {sentences_total}")
    lines.append("")
    
    # Path B: Skeleton stats (always present)
    lines.append("## Path B: A1 Skeleton Builder (No LLM)")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Sentences processed | {path_b_stats['processed']} |")
    lines.append(f"| Avg tokens/sentence | {path_b_stats.get('avg_tokens', 0):.1f} |")
    lines.append(f"| Avg A1 coverage | {path_b_stats.get('avg_coverage', 0):.1%} |")
    lines.append(f"| Avg atoms/skeleton | {path_b_stats.get('avg_atoms', 0):.1f} |")
    lines.append(f"| Zero-coverage sentences | {path_b_stats.get('zero_coverage', 0)} |")
    lines.append("")
    
    # Top atoms across all skeletons
    if path_b_stats.get("atom_frequency"):
        lines.append("### Top 30 Atoms (by frequency across skeletons)")
        lines.append("")
        lines.append("| Atom | Count |")
        lines.append("|------|-------|")
        for atom, count in path_b_stats["atom_frequency"].most_common(30):
            lines.append(f"| {atom} | {count} |")
        lines.append("")
    
    # Category distribution
    if path_b_stats.get("category_dist"):
        lines.append("### Category Distribution")
        lines.append("")
        lines.append("| Category | Count | % |")
        lines.append("|----------|-------|---|")
        total_cat = sum(path_b_stats["category_dist"].values())
        for cat, count in sorted(path_b_stats["category_dist"].items(),
                                  key=lambda x: -x[1]):
            pct = count / total_cat * 100 if total_cat > 0 else 0
            lines.append(f"| {cat} | {count} | {pct:.1f}% |")
        lines.append("")
    
    # Path A: Pipeline stats (if available)
    if path_a_stats:
        lines.append("## Path A: Current Pipeline (LLM-based)")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Processed | {path_a_stats.get('processed', 0)} |")
        lines.append(f"| Success | {path_a_stats.get('success', 0)} |")
        lines.append(f"| Failed | {path_a_stats.get('failed', 0)} |")
        lines.append(f"| Abstained (no candidates) | {path_a_stats.get('abstained', 0)} |")
        
        if path_a_stats.get('processed', 0) > 0:
            success_rate = path_a_stats.get('success', 0) / path_a_stats['processed'] * 100
            lines.append(f"| **Success rate** | **{success_rate:.1f}%** |")
        lines.append("")
        
        # Formula patterns
        if path_a_stats.get("formula_patterns"):
            lines.append("### Formula Patterns (top 20)")
            lines.append("")
            lines.append("| Pattern | Count |")
            lines.append("|---------|-------|")
            for pattern, count in path_a_stats["formula_patterns"].most_common(20):
                lines.append(f"| `{pattern}` | {count} |")
            lines.append("")
    else:
        lines.append("## Path A: Current Pipeline")
        lines.append("")
        lines.append("*Skipped (--skeleton-only or LLM unavailable)*")
        lines.append("")
    
    # Comparison (if both paths ran)
    if path_a_stats and path_a_stats.get("success", 0) > 0:
        lines.append("## Path A vs Path B Comparison")
        lines.append("")
        lines.append("*To be populated after inspection of generated molecules.*")
        lines.append("")
    
    # Next steps
    lines.append("## Observations & Next Steps")
    lines.append("")
    lines.append("*(To be filled by Taka after reviewing output files)*")
    lines.append("")
    
    report_path = os.path.join(out_dir, "pilot_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✅ Report: {report_path}")
    
    # Stats JSON
    stats = {
        "dataset": dataset_name,
        "articles": articles_processed,
        "sentences": sentences_total,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "path_b": {
            k: v for k, v in path_b_stats.items()
            if k not in ("atom_frequency", "category_dist")
        },
    }
    if path_b_stats.get("atom_frequency"):
        stats["path_b"]["top_atoms"] = dict(path_b_stats["atom_frequency"].most_common(50))
    if path_b_stats.get("category_dist"):
        stats["path_b"]["category_dist"] = dict(path_b_stats["category_dist"])
    if path_a_stats:
        stats["path_a"] = {
            k: v for k, v in path_a_stats.items()
            if k != "formula_patterns"
        }
        if path_a_stats.get("formula_patterns"):
            stats["path_a"]["top_formulas"] = dict(path_a_stats["formula_patterns"].most_common(30))
    
    stats_path = os.path.join(out_dir, "pilot_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Stats: {stats_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="ESDE Phase 8 — Molecule Generation Pilot (Option C)"
    )
    parser.add_argument("--dataset", default="mixed",
                        help="Dataset name (default: mixed)")
    parser.add_argument("--max-articles", type=int, default=0,
                        help="Max articles to process (0 = all)")
    parser.add_argument("--max-sentences", type=int, default=0,
                        help="Max sentences per article (0 = all)")
    parser.add_argument("--skeleton-only", action="store_true",
                        help="Run Path B only (no LLM needed)")
    parser.add_argument("--a1-centroids", default="integration/lexicon/synapse_v4_report/atom_centroids_48d_raw.csv",
                        help="Path to atom centroids CSV")
    parser.add_argument("--a1-dir", default="integration/lexicon/audit_output/",
                        help="Path to A1 final JSONL directory")
    parser.add_argument("--synapse", default="esde_synapses_v3.json",
                        help="Synapse v3 base file")
    parser.add_argument("--glossary", default="sensor/glossary.json",
                        help="Glossary file for Sensor/Generator")
    parser.add_argument("--synapse-patches", nargs="*", default=[],
                        help="Synapse patch files")
    parser.add_argument("--out-dir", default="output/phase8_pilot",
                        help="Output directory")
    args = parser.parse_args()
    
    print("=" * 70)
    print("  ESDE Phase 8 — Molecule Generation Pilot")
    print("  Option C: Pipeline + A1 Skeleton in parallel")
    print("=" * 70)
    
    # ── Load Harvest Data ──
    print(f"\n[1/4] Loading dataset '{args.dataset}'...")
    try:
        from harvester.storage import load_dataset
        articles = load_dataset(args.dataset)
    except (ImportError, FileNotFoundError) as e:
        print(f"  ❌ {e}")
        print(f"  Run: python -m harvester.cli harvest --dataset {args.dataset}")
        sys.exit(1)
    
    article_ids = sorted(articles.keys())
    if args.max_articles > 0:
        article_ids = article_ids[:args.max_articles]
    
    print(f"  Articles available: {len(articles)}")
    print(f"  Articles to process: {len(article_ids)}")
    
    # ── Initialize Path B: Skeleton Builder ──
    print(f"\n[2/4] Initializing A1 Skeleton Builder...")
    try:
        skeleton_builder = A1SkeletonBuilder(
            centroids_path=args.a1_centroids,
            a1_dir=args.a1_dir,
        )
    except FileNotFoundError as e:
        print(f"  ❌ {e}")
        print(f"  Run synapse_v4_compare.py first to generate centroids.")
        sys.exit(1)
    
    # ── Initialize Path A: Pipeline (optional) ──
    pipeline_runner = None
    if not args.skeleton_only:
        print(f"\n[3/4] Initializing Pipeline (Path A)...")
        
        if not check_llm_available():
            print(f"  ⚠ LLM server not available. Path A will be skipped.")
            print(f"  (Use --skeleton-only to suppress this warning)")
        else:
            try:
                pipeline_runner = PipelineRunner(
                    synapse_file=args.synapse,
                    glossary_file=args.glossary,
                    synapse_patches=args.synapse_patches,
                )
                print(f"  ✅ Pipeline ready (Sensor + Generator + Validator)")
            except Exception as e:
                print(f"  ⚠ Pipeline init failed: {e}")
                print(f"  Path A will be skipped.")
    else:
        print(f"\n[3/4] Pipeline skipped (--skeleton-only)")
    
    # ── Process Articles ──
    print(f"\n[4/4] Processing articles...")
    
    out_base = os.path.join(args.out_dir, args.dataset)
    mol_dir = os.path.join(out_base, "molecules")
    skel_dir = os.path.join(out_base, "skeletons")
    os.makedirs(mol_dir, exist_ok=True)
    os.makedirs(skel_dir, exist_ok=True)
    
    # Aggregate stats
    total_sentences = 0
    path_b_stats = {
        "processed": 0, "avg_tokens": 0, "avg_coverage": 0,
        "avg_atoms": 0, "zero_coverage": 0,
        "atom_frequency": Counter(), "category_dist": Counter(),
    }
    path_a_stats_agg = None
    if pipeline_runner:
        path_a_stats_agg = {
            "processed": 0, "success": 0, "failed": 0, "abstained": 0,
            "formula_patterns": Counter(),
        }
    
    _tokens_sum = 0
    _coverage_sum = 0.0
    _atoms_sum = 0
    
    for article_idx, article_id in enumerate(article_ids, 1):
        text = articles[article_id]
        sentences = segment_sentences(text)
        
        if args.max_sentences > 0:
            sentences = sentences[:args.max_sentences]
        
        total_sentences += len(sentences)
        
        print(f"\n  [{article_idx}/{len(article_ids)}] {article_id}: "
              f"{len(sentences)} sentences")
        
        # Path B: Skeletons
        skel_path = os.path.join(skel_dir, f"{article_id}_skeletons.jsonl")
        with open(skel_path, "w") as sf:
            for sent_idx, sentence in enumerate(sentences):
                skeleton = skeleton_builder.build_skeleton(sentence)
                skeleton["article_id"] = article_id
                skeleton["sentence_idx"] = sent_idx
                sf.write(json.dumps(skeleton, ensure_ascii=False) + "\n")
                
                # Accumulate stats
                path_b_stats["processed"] += 1
                _tokens_sum += skeleton["tokens_total"]
                _coverage_sum += skeleton["coverage"]
                _atoms_sum += len(skeleton["active_atoms"])
                
                if skeleton["coverage"] == 0:
                    path_b_stats["zero_coverage"] += 1
                
                for aa in skeleton["active_atoms"]:
                    atom = aa["atom"]
                    path_b_stats["atom_frequency"][atom] += 1
                    cat = atom.split(".")[0] if "." in atom else "?"
                    path_b_stats["category_dist"][cat] += 1
        
        # Path A: Pipeline molecules (if available)
        if pipeline_runner:
            mol_path = os.path.join(mol_dir, f"{article_id}_molecules.jsonl")
            with open(mol_path, "w") as mf:
                for sent_idx, sentence in enumerate(sentences):
                    result = pipeline_runner.process_sentence(sentence)
                    if result:
                        result["article_id"] = article_id
                        result["sentence_idx"] = sent_idx
                        mf.write(json.dumps(result, ensure_ascii=False) + "\n")
                        
                        # Track formula patterns
                        formula = result.get("formula", "")
                        if formula:
                            # Normalize: replace atom IDs with placeholders
                            pattern = re.sub(r'aa_\d+', 'X', formula)
                            path_a_stats_agg["formula_patterns"][pattern] += 1
                    
                    # Progress
                    if (sent_idx + 1) % 50 == 0:
                        print(f"    Path A: {sent_idx + 1}/{len(sentences)}")
            
            # Merge pipeline stats
            path_a_stats_agg["processed"] += pipeline_runner.stats["processed"]
            path_a_stats_agg["success"] += pipeline_runner.stats["success"]
            path_a_stats_agg["failed"] += pipeline_runner.stats["failed"]
            path_a_stats_agg["abstained"] += pipeline_runner.stats["abstained"]
            # Reset per-article
            pipeline_runner.stats = {"processed": 0, "success": 0, "failed": 0, "abstained": 0}
    
    # Compute averages
    n = path_b_stats["processed"]
    if n > 0:
        path_b_stats["avg_tokens"] = _tokens_sum / n
        path_b_stats["avg_coverage"] = _coverage_sum / n
        path_b_stats["avg_atoms"] = _atoms_sum / n
    
    # ── Generate Report ──
    print("\n" + "=" * 70)
    print("  Generating Report")
    print("=" * 70)
    
    generate_report(
        dataset_name=args.dataset,
        articles_processed=len(article_ids),
        sentences_total=total_sentences,
        path_a_stats=path_a_stats_agg,
        path_b_stats=path_b_stats,
        out_dir=out_base,
    )
    
    # ── Summary ──
    print("\n" + "=" * 70)
    print("  PILOT COMPLETE")
    print("=" * 70)
    print(f"  Articles: {len(article_ids)}")
    print(f"  Sentences: {total_sentences}")
    print(f"  Path B (skeleton): {path_b_stats['processed']} processed, "
          f"avg coverage {path_b_stats['avg_coverage']:.1%}")
    
    if path_a_stats_agg:
        sr = (path_a_stats_agg['success'] / path_a_stats_agg['processed'] * 100
              if path_a_stats_agg['processed'] > 0 else 0)
        print(f"  Path A (pipeline): {path_a_stats_agg['processed']} processed, "
              f"success rate {sr:.1f}%")
    
    print(f"\n  Output: {out_base}/")


if __name__ == "__main__":
    main()
