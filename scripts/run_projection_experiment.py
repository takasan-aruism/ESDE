#!/usr/bin/env python3
"""
ESDE Phase 8 — Projection Operator Experiment Runner
=====================================================
Runs Base/B/C/BC modes on a sentence set and produces predictions
for evaluation by tools/projection_eval.py.

Usage:
  # Single mode
  python3 scripts/run_projection_experiment.py \
    --mode BC \
    --sentences output/projection_eval/berlin_sentences.jsonl \
    --dictionary lexicon_wn/esde_dictionary.json \
    --synapse esde_synapses_v3.json \
    --gt output/projection_eval/ground_truth_50.jsonl \
    --out output/projection_eval/BC/

  # All modes (comparison)
  python3 scripts/run_projection_experiment.py \
    --mode all \
    --sentences output/projection_eval/berlin_sentences.jsonl \
    --dictionary lexicon_wn/esde_dictionary.json \
    --synapse esde_synapses_v3.json \
    --gt output/projection_eval/ground_truth_50.jsonl \
    --out output/projection_eval/

Sentence JSONL schema:
{
  "id": "berlin_0001",
  "sentence": "Berlin is the capital and largest city of Germany.",
  "tokens": [...]   // optional; will be tokenized if absent
}

Author: Claude (Implementation) per GPT spec §6
Date: 2026-03-03
"""

import json
import argparse
import time
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

import numpy as np

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from esde.projection import (
    get_embedder, AtomFieldEmbeddings,
    FieldFirstProjection, WeakMeasurementProjection, HybridProjection,
    get_projection_operator, build_prior_from_synapse_candidates,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ============================================================
# Minimal Synapse lookup (standalone, no full sensor import)
# ============================================================

class SimpleSynapseLoader:
    """
    Lightweight Synapse loader for experiment runner.
    Loads synapse JSON and provides word→atom candidate lookup.
    """
    
    def __init__(self, synapse_path: str, patches: List[str] = None):
        self.synapse_path = synapse_path
        self.synapses: Dict[str, List[Dict]] = {}
        self._load(patches or [])
    
    def _load(self, patches: List[str]):
        with open(self.synapse_path) as f:
            data = json.load(f)
        self.synapses = data.get("synapses", {})
        
        # Apply patches (overlay)
        for patch_path in patches:
            with open(patch_path) as f:
                patch_data = json.load(f)
            patch_list = patch_data.get("patches", [])
            for entry in patch_list:
                synset_id = entry.get("synset_id", "")
                if entry.get("disable_edge"):
                    # Tombstone
                    self.synapses.pop(synset_id, None)
                else:
                    if synset_id not in self.synapses:
                        self.synapses[synset_id] = []
                    self.synapses[synset_id].append(entry)
        
        logger.info(f"Synapse loaded: {len(self.synapses)} synsets")
    
    def get_edges(self, synset_id: str) -> List[Dict]:
        return self.synapses.get(synset_id, [])


class SimpleTokenProcessor:
    """
    Minimal token→atom candidate processor.
    Uses NLTK WordNet for synset extraction + Synapse lookup.
    """
    
    # POS filter matching rule_generator
    ALLOWED_POS = {"NOUN", "VERB", "ADJ", "PROPN"}
    
    # Stopwords matching rule_generator
    STOPWORDS = {
        "be", "is", "was", "were", "been", "being", "am", "are",
        "have", "has", "had", "having",
        "do", "does", "did", "doing",
        "get", "got", "getting",
        "make", "made", "making",
        "go", "went", "gone", "going",
        "come", "came", "coming",
        "take", "took", "taken", "taking",
        "give", "gave", "given", "giving",
        "say", "said", "saying",
        "know", "knew", "known", "knowing",
        "think", "thought", "thinking",
        "see", "saw", "seen", "seeing",
        "want", "wanted", "wanting",
        "use", "used", "using",
        "find", "found", "finding",
        "tell", "told", "telling",
        "may", "might", "can", "could", "will", "would", "shall", "should",
        "well", "also", "just", "new", "old", "good", "bad", "great",
        "many", "much", "more", "most", "other", "some", "such",
        "first", "last", "long", "large", "small", "high", "low",
        "early", "late", "young",
    }
    
    # spaCy POS → WordNet POS
    POS_MAP = {"NOUN": "n", "VERB": "v", "ADJ": "a", "PROPN": "n"}
    
    def __init__(self, synapse: SimpleSynapseLoader, min_score: float = 0.45):
        self.synapse = synapse
        self.min_score = min_score
        
        # Try to import WordNet
        try:
            from nltk.corpus import wordnet as wn
            self.wn = wn
            wn.synsets('test')  # trigger load
        except Exception:
            logger.warning("NLTK WordNet not available. Install with: pip install nltk")
            self.wn = None
    
    def get_candidates(self, word: str, pos: str) -> List[Dict]:
        """
        Get Synapse atom candidates for a word.
        
        Returns: [{"atom": "STA.wealth", "score": 0.72}, ...]
        """
        if pos not in self.ALLOWED_POS:
            return []
        if word.lower() in self.STOPWORDS:
            return []
        
        if self.wn is None:
            return []
        
        # Get WordNet synsets
        wn_pos = self.POS_MAP.get(pos)
        lemma = word.lower()
        synsets = self.wn.synsets(lemma, pos=wn_pos) if wn_pos else self.wn.synsets(lemma)
        
        # Lookup in Synapse
        candidates = {}
        for ss in synsets:
            synset_id = ss.name()
            edges = self.synapse.get_edges(synset_id)
            for edge in edges:
                atom_id = edge.get("concept_id", "")
                score = edge.get("raw_score", 0.0) * edge.get("weight", 1.0)
                if atom_id and score >= self.min_score:
                    if atom_id not in candidates or score > candidates[atom_id]:
                        candidates[atom_id] = score
        
        # Sort descending
        sorted_cands = sorted(candidates.items(), key=lambda x: (-x[1], x[0]))
        return [{"atom": a, "score": s} for a, s in sorted_cands]


# ============================================================
# Experiment Runner
# ============================================================

def tokenize_sentence(sentence: str) -> List[Dict]:
    """
    Simple tokenization with POS tagging.
    Uses NLTK POS tagger if spaCy unavailable.
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(sentence)
        return [{"text": tok.text, "lemma": tok.lemma_, "pos": tok.pos_, "i": tok.i}
                for tok in doc]
    except Exception:
        pass
    
    # Fallback: NLTK
    try:
        import nltk
        from nltk import word_tokenize, pos_tag
        tokens = word_tokenize(sentence)
        tagged = pos_tag(tokens)
        
        # Map Penn Treebank tags to Universal POS
        ptb_to_upos = {
            "NN": "NOUN", "NNS": "NOUN", "NNP": "PROPN", "NNPS": "PROPN",
            "VB": "VERB", "VBD": "VERB", "VBG": "VERB", "VBN": "VERB",
            "VBP": "VERB", "VBZ": "VERB",
            "JJ": "ADJ", "JJR": "ADJ", "JJS": "ADJ",
            "RB": "ADV", "RBR": "ADV", "RBS": "ADV",
        }
        result = []
        for i, (tok, tag) in enumerate(tagged):
            upos = ptb_to_upos.get(tag, "X")
            result.append({"text": tok, "lemma": tok.lower(), "pos": upos, "i": i})
        return result
    except Exception:
        # Ultra-fallback: split only, all NOUN
        return [{"text": w, "lemma": w.lower(), "pos": "NOUN", "i": i}
                for i, w in enumerate(sentence.split())]


def run_single_sentence(sentence_rec: dict, processor: SimpleTokenProcessor,
                        operator, embedder, atom_field: AtomFieldEmbeddings,
                        mode: str) -> dict:
    """
    Run projection on a single sentence.
    
    Returns per-word prediction records.
    """
    sentence = sentence_rec["sentence"]
    tokens = tokenize_sentence(sentence)
    
    # Embed sentence once
    if mode != "base" and embedder is not None:
        sent_emb = embedder.embed([sentence])[0]
    else:
        sent_emb = None
    
    targets = []
    for tok in tokens:
        word = tok["text"]
        pos = tok["pos"]
        
        # Get Synapse candidates
        synapse_cands = processor.get_candidates(word, pos)
        if not synapse_cands:
            continue
        
        # Apply projection
        if mode == "base" or operator is None or sent_emb is None:
            # No projection — use raw Synapse
            proj_top3 = synapse_cands[:3]
            gate_stats = {}
        else:
            proj_result, gate_stats = operator.project(
                synapse_candidates=synapse_cands,
                sentence_emb=sent_emb,
                atom_field=atom_field,
                top_k=5,
            )
            proj_top3 = proj_result[:3]
        
        targets.append({
            "span_text": word,
            "pos": pos,
            "synapse_topk": [{"atom": c["atom"], "score": round(c["score"], 4)}
                             for c in synapse_cands[:5]],
            "pred_top3": [c["atom"] for c in proj_top3],
            "scores_top3": [round(c["score"], 4) for c in proj_top3],
            "gate_stats": gate_stats,
        })
    
    return {
        "id": sentence_rec["id"],
        "sentence": sentence,
        "targets": targets,
    }


def run_experiment(mode: str, sentences: List[dict],
                   processor: SimpleTokenProcessor,
                   dictionary_path: str,
                   embedder_backend: str = "tfidf",
                   out_dir: str = "output/projection_eval") -> dict:
    """
    Run experiment for a single mode.
    
    Returns: {mode, results, timing, stats}
    """
    out_path = Path(out_dir) / mode
    out_path.mkdir(parents=True, exist_ok=True)
    
    t0 = time.time()
    
    # Setup embedder and atom field
    if mode != "base":
        embedder = get_embedder(backend=embedder_backend)
        
        # For TF-IDF: fit on atom definitions + all sentences
        if hasattr(embedder, 'fit') and hasattr(embedder, '_fitted') and not embedder._fitted:
            with open(dictionary_path) as f:
                raw = json.load(f)
            concepts = raw.get("concepts", raw)
            if isinstance(concepts, dict) and "atoms" in concepts:
                concepts = concepts["atoms"]
            
            corpus = []
            for atom_id, entry in concepts.items():
                parts = [atom_id]
                if "en_label" in entry:
                    parts.append(entry["en_label"])
                if "short_definition" in entry:
                    parts.append(entry["short_definition"])
                corpus.append(". ".join(parts))
            
            for s in sentences:
                corpus.append(s["sentence"])
            
            embedder.fit(corpus)
        
        atom_field = AtomFieldEmbeddings(dictionary_path, embedder)
        operator = get_projection_operator(mode)
    else:
        embedder = None
        atom_field = None
        operator = None
    
    t_setup = time.time() - t0
    
    # Run per sentence
    results = []
    gate_stats_all = []
    
    for sent_rec in sentences:
        result = run_single_sentence(
            sent_rec, processor, operator, embedder, atom_field, mode
        )
        results.append(result)
        
        for t in result.get("targets", []):
            if t.get("gate_stats"):
                gate_stats_all.append(t["gate_stats"])
    
    t_total = time.time() - t0
    
    # Write pred JSONL (scorer-compatible format)
    pred_path = out_path / "pred_50.jsonl"
    with open(pred_path, 'w') as f:
        for r in results:
            pred_rec = {
                "id": r["id"],
                "targets": [
                    {
                        "span_text": t["span_text"],
                        "pred_top3": t["pred_top3"],
                        "scores_top3": t["scores_top3"],
                    }
                    for t in r["targets"]
                ],
            }
            f.write(json.dumps(pred_rec, ensure_ascii=False) + "\n")
    
    # Write detailed results
    detail_path = out_path / "detail.jsonl"
    with open(detail_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    # Write field stats
    if gate_stats_all:
        tau_fields = [s.get("tau_field", 0) for s in gate_stats_all if "tau_field" in s]
        g_maxes = [s.get("g_max", 0) for s in gate_stats_all if "g_max" in s]
        
        field_stats = {
            "mode": mode,
            "n_sentences": len(results),
            "n_targets_with_gate": len(gate_stats_all),
            "tau_field_mean": float(np.mean(tau_fields)) if tau_fields else None,
            "tau_field_std": float(np.std(tau_fields)) if tau_fields else None,
            "g_max_mean": float(np.mean(g_maxes)) if g_maxes else None,
            "g_max_std": float(np.std(g_maxes)) if g_maxes else None,
        }
        with open(out_path / "field_stats.json", 'w') as f:
            json.dump(field_stats, f, indent=2)
    
    timing = {
        "setup_sec": round(t_setup, 2),
        "total_sec": round(t_total, 2),
        "per_sentence_ms": round((t_total - t_setup) / max(len(sentences), 1) * 1000, 1),
    }
    
    logger.info(f"[{mode}] Done: {len(results)} sentences, {timing['total_sec']}s total, "
                f"{timing['per_sentence_ms']}ms/sent")
    
    return {
        "mode": mode,
        "n_sentences": len(results),
        "n_targets": sum(len(r["targets"]) for r in results),
        "timing": timing,
        "pred_path": str(pred_path),
    }


def generate_report(all_results: Dict[str, dict], gt_path: str, out_dir: str):
    """
    Generate comparison report.md using projection_eval scorer.
    """
    # Import scorer
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    from projection_eval import evaluate
    
    report_lines = [
        "# ESDE Projection Operator — Experiment Report",
        f"Date: {time.strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Results Summary",
        "",
        "| Mode | Recall@1 | Recall@3 | Targets | Time (s) | ms/sent |",
        "|------|----------|----------|---------|----------|---------|",
    ]
    
    mode_evals = {}
    for mode in ["base", "B", "C", "BC"]:
        info = all_results.get(mode)
        if not info:
            continue
        
        pred_path = info["pred_path"]
        if os.path.exists(pred_path) and os.path.exists(gt_path):
            eval_result = evaluate(gt_path, pred_path)
            mode_evals[mode] = eval_result
            r1 = eval_result["recall_at_1"]
            r3 = eval_result["recall_at_3"]
            n_eval_targets = eval_result["n_targets"]
        else:
            r1, r3 = 0.0, 0.0
            n_eval_targets = info['n_targets']
        
        t = info["timing"]
        report_lines.append(
            f"| {mode:<4} | {r1:.3f}    | {r3:.3f}    | {n_eval_targets:<7} | "
            f"{t['total_sec']:<8} | {t['per_sentence_ms']:<7} |"
        )
    
    report_lines.extend(["", "## Timing Detail", ""])
    for mode, info in all_results.items():
        t = info["timing"]
        report_lines.append(f"- **{mode}**: setup={t['setup_sec']}s, total={t['total_sec']}s")
    
    # Confusion analysis for best mode
    best_mode = max(mode_evals, key=lambda m: mode_evals[m]["recall_at_3"]) if mode_evals else None
    if best_mode:
        ev = mode_evals[best_mode]
        report_lines.extend([
            "",
            f"## Confusion Analysis (best: {best_mode}, R@3={ev['recall_at_3']:.3f})",
            "",
        ])
        for c in ev["confusions"][:15]:
            report_lines.append(
                f"- `{c['span_text']}`: pred={c['pred_top1']} → gt={c['gt_atoms']}"
            )
    
    report_lines.extend(["", "---", "*記述せよ、しかし決定するな*"])
    
    report_path = Path(out_dir) / "report.md"
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
    
    logger.info(f"Report written: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="ESDE Projection Experiment Runner")
    parser.add_argument("--mode", required=True,
                        help="base|B|C|BC|all")
    parser.add_argument("--sentences", required=True,
                        help="Sentence JSONL (berlin_sentences.jsonl)")
    parser.add_argument("--dictionary", required=True,
                        help="esde_dictionary.json path")
    parser.add_argument("--synapse", required=True,
                        help="esde_synapses_v3.json path")
    parser.add_argument("--synapse-patches", nargs="*", default=[],
                        help="Synapse patch files")
    parser.add_argument("--gt", default=None,
                        help="Ground truth JSONL (for report generation)")
    parser.add_argument("--out", default="output/projection_eval",
                        help="Output directory")
    parser.add_argument("--embedder", default="tfidf",
                        choices=["tfidf", "minilm"],
                        help="Embedding backend")
    args = parser.parse_args()
    
    # Load sentences
    sentences = []
    with open(args.sentences) as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(json.loads(line))
    logger.info(f"Loaded {len(sentences)} sentences")
    
    # Load Synapse
    synapse = SimpleSynapseLoader(args.synapse, args.synapse_patches)
    processor = SimpleTokenProcessor(synapse)
    
    # Determine modes
    if args.mode == "all":
        modes = ["base", "B", "C", "BC"]
    else:
        modes = [args.mode]
    
    # Run
    all_results = {}
    for mode in modes:
        logger.info(f"--- Running mode: {mode} ---")
        result = run_experiment(
            mode=mode,
            sentences=sentences,
            processor=processor,
            dictionary_path=args.dictionary,
            embedder_backend=args.embedder,
            out_dir=args.out,
        )
        all_results[mode] = result
    
    # Generate comparison report
    if args.gt and len(modes) > 1:
        generate_report(all_results, args.gt, args.out)
    elif args.gt and len(modes) == 1:
        # Single mode evaluation
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
        from projection_eval import evaluate, print_report
        ev = evaluate(args.gt, all_results[modes[0]]["pred_path"])
        print_report(ev, mode=modes[0])
    
    print("\nDone. Results in:", args.out)


if __name__ == "__main__":
    main()