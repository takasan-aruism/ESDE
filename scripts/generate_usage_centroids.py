#!/usr/bin/env python3
"""
ESDE Phase 8.5 — Usage-Based Atom Field (UBAF) Centroid Generator
=================================================================
Reconstructs atom centroids from corpus usage distributions.

Spec: GPT Audit — ESDE_UBAF_Implementation_Instructions.txt
Implementation: Claude

Pipeline:
  1) Load dictionary + existing centroids C_old (MiniLM 384D)
  2) Load Synapse v3 (word→atom scores)
  3) Load A1 word 48D vectors + atom 48D centroids → cross-resonance
  4) Scan corpus: extract ±8 context windows per target word
  5) Embed contexts with MiniLM (384D)
  6) Compute weighted centroids C_new
  7) Dynamic λ blending: C_final = normalize((1-λ)*C_old + λ*C_new)
  8) Generate drift report + save

Usage:
  python3 scripts/generate_usage_centroids.py \\
    --dictionary integration/lexicon/esde_dictionary.json \\
    --synapse esde_synapses_v3.json \\
    --corpus data/datasets/mixed/ \\
    --a1-final-dir integration/lexicon/audit_output/ \\
    --a1-centroids integration/lexicon/synapse_v4_report/atom_centroids_48d.csv \\
    --out output/ubaf/

Author: Claude (Implementation) per GPT Audit Spec
Date: 2026-03-03
"""

import json
import csv
import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ============================================================
# Constants (GPT Spec)
# ============================================================

PROTOTYPE_ATOMS = [
    "SOC.official", "SPC.place", "STA.wealth", "SOC.city",
    "SOC.nation", "STA.war", "PRP.part", "WLD.culture",
    "ACT.descend", "SOC.public",
]

CONTEXT_WINDOW = 8       # ±8 tokens (§2)
MIN_CONTEXT_TOKENS = 3   # (§2)
P_EXPONENT = 1           # v3^p (§3)
Q_EXPONENT = 1           # a1^q (§3)
SUPPORT_DENOMINATOR = 2000  # (§7, adjusted per GPT)

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "must",
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their",
    "this", "that", "these", "those", "which", "who", "whom", "what",
    "and", "or", "but", "if", "then", "than", "when", "where", "while",
    "as", "at", "by", "for", "from", "in", "into", "of", "on", "to",
    "with", "not", "no", "so", "very", "just", "also", "more", "most",
    "about", "after", "before", "between", "both", "each", "every",
    "other", "such", "only", "own", "same", "too",
}


# ============================================================
# A1 48D Data Loaders
# ============================================================

def load_a1_word_vectors(a1_final_dir: str, target_atoms: List[str]) -> Dict[str, np.ndarray]:
    """
    Load word → 48D normalized_scores from audit_output/*_a1_final.jsonl.

    Returns: {"capital": np.array([48 floats]), "wealth": ..., ...}
    Note: each word appears only in its assigned atom's file.
    """
    word_vectors: Dict[str, np.ndarray] = {}
    slot_order = None  # Will be set from first record

    for atom_id in target_atoms:
        # SOC.official → SOC_official
        file_stem = atom_id.replace(".", "_")
        path = Path(a1_final_dir) / f"{file_stem}_a1_final.jsonl"

        if not path.exists():
            logger.warning(f"A1 final not found: {path}")
            continue

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                word = rec.get("word", "").lower()
                ns = rec.get("normalized_scores", {})

                if not word or not ns:
                    continue

                # Establish slot order from first record
                if slot_order is None:
                    slot_order = sorted(ns.keys())

                vec = np.array([ns.get(s, 0.0) for s in slot_order], dtype=np.float64)
                word_vectors[word] = vec

    logger.info(f"Loaded A1 word vectors: {len(word_vectors)} words, {len(slot_order) if slot_order else 0}D")
    return word_vectors


def load_a1_atom_centroids(csv_path: str, target_atoms: List[str]) -> Dict[str, np.ndarray]:
    """
    Load atom → 48D centroid from atom_centroids_48d.csv.
    """
    centroids = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        # First field is atom_id, rest are dimensions
        dim_fields = fields[1:]

        for row in reader:
            atom_id = row[fields[0]].strip()
            if atom_id in set(target_atoms):
                vec = np.array([float(row[d]) for d in dim_fields], dtype=np.float64)
                centroids[atom_id] = vec

    logger.info(f"Loaded A1 atom centroids: {len(centroids)} atoms, {len(dim_fields)}D")
    return centroids


def compute_a1_resonance(
    word: str,
    atom_id: str,
    word_vectors: Dict[str, np.ndarray],
    atom_centroids_48d: Dict[str, np.ndarray],
) -> float:
    """
    A1_Resonance(w, A) = cos(x_w, x_A) in 48D space.
    Returns 0.0 if either vector is missing.
    """
    w_vec = word_vectors.get(word.lower())
    a_vec = atom_centroids_48d.get(atom_id)

    if w_vec is None or a_vec is None:
        return 0.0

    norm_w = np.linalg.norm(w_vec)
    norm_a = np.linalg.norm(a_vec)

    if norm_w < 1e-10 or norm_a < 1e-10:
        return 0.0

    return float(np.dot(w_vec, a_vec) / (norm_w * norm_a))


# ============================================================
# Synapse v3 Loader
# ============================================================

def load_synapse_v3(path: str, target_atoms: Set[str]) -> Dict[str, Dict[str, float]]:
    """
    Load Synapse v3 → build atom → {word: max_score} index.
    Only for target atoms.
    """
    with open(path) as f:
        data = json.load(f)

    synapses = data.get("synapses", data)

    # atom → {lemma: max_score}
    atom_words: Dict[str, Dict[str, float]] = defaultdict(dict)

    for synset_id, candidates in synapses.items():
        parts = synset_id.split(".")
        lemma = parts[0].replace("_", " ").lower() if len(parts) >= 2 else synset_id.lower()

        for cand in candidates:
            atom = cand.get("concept_id", "")
            score = cand.get("weight", 0.0)
            if atom in target_atoms and score > 0:
                atom_words[atom][lemma] = max(atom_words[atom].get(lemma, 0.0), score)

    total = sum(len(v) for v in atom_words.values())
    logger.info(f"Loaded Synapse v3: {total} word→atom mappings for {len(atom_words)} target atoms")
    return dict(atom_words)


# ============================================================
# Corpus + Context
# ============================================================

def load_corpus(corpus_dir: str) -> List[Tuple[str, str]]:
    """Load city_*.txt → [(article_id, sentence), ...]."""
    sentences = []
    for txt in sorted(Path(corpus_dir).glob("city_*.txt")):
        article = txt.stem
        text = txt.read_text(encoding="utf-8")
        for s in re.split(r'(?<=[.!?])\s+', text):
            s = s.strip()
            if len(s) > 20:
                sentences.append((article, s))
    logger.info(f"Corpus: {len(sentences)} sentences from {corpus_dir}")
    return sentences


def tokenize(sentence: str) -> List[str]:
    tokens = sentence.split()
    return [re.sub(r'^[^\w]+|[^\w]+$', '', t) for t in tokens if re.sub(r'^[^\w]+|[^\w]+$', '', t)]


def extract_context(tokens: List[str], idx: int, window: int = CONTEXT_WINDOW) -> Optional[str]:
    """±window tokens, stopwords removed. Returns None if < MIN_CONTEXT_TOKENS."""
    start = max(0, idx - window)
    end = min(len(tokens), idx + window + 1)
    filtered = [t for t in tokens[start:end] if t.lower() not in STOPWORDS and len(t) > 1]
    if len(filtered) < MIN_CONTEXT_TOKENS:
        return None
    return " ".join(filtered)


# ============================================================
# Main Pipeline
# ============================================================

def run(args):
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    target_atoms = PROTOTYPE_ATOMS
    target_set = set(target_atoms)

    # ----------------------------------------------------------
    # 1. Load existing MiniLM centroids (C_old, 384D)
    # ----------------------------------------------------------
    logger.info("=== Step 1: Load C_old (MiniLM 384D) ===")
    from esde.projection import get_embedder, AtomFieldEmbeddings

    embedder = get_embedder(args.embedder)
    atom_field = AtomFieldEmbeddings(args.dictionary, embedder)

    C_old = {}
    for a in target_atoms:
        idx = atom_field._atom_to_idx.get(a)
        if idx is not None:
            C_old[a] = atom_field.Z[idx].copy()
    logger.info(f"C_old: {len(C_old)} atoms, dim={atom_field.Z.shape[1]}")

    # ----------------------------------------------------------
    # 2. Load Synapse v3
    # ----------------------------------------------------------
    logger.info("=== Step 2: Load Synapse v3 ===")
    atom_words = load_synapse_v3(args.synapse, target_set)

    for a in target_atoms:
        n = len(atom_words.get(a, {}))
        logger.info(f"  {a}: {n} words in Synapse")

    # ----------------------------------------------------------
    # 3. Load A1 48D data
    # ----------------------------------------------------------
    logger.info("=== Step 3: Load A1 48D data ===")

    # Load ALL atom finals (not just target) because words live in their assigned atom file
    # but we need to compute cross-atom resonance
    all_atom_ids = list(atom_field._atom_to_idx.keys())
    word_vectors_48d = load_a1_word_vectors(args.a1_final_dir, all_atom_ids)
    atom_centroids_48d = load_a1_atom_centroids(args.a1_centroids, target_atoms)

    # Quick check: how many Synapse words have A1 vectors?
    for a in target_atoms:
        words = atom_words.get(a, {})
        has_a1 = sum(1 for w in words if w in word_vectors_48d)
        logger.info(f"  {a}: {has_a1}/{len(words)} Synapse words have A1 vectors")

    # ----------------------------------------------------------
    # 4. Scan corpus
    # ----------------------------------------------------------
    logger.info("=== Step 4: Scan corpus ===")
    sentences = load_corpus(args.corpus)
    N = len(sentences)

    # All words that map to any target atom
    all_target_words = set()
    for a in target_atoms:
        all_target_words.update(atom_words.get(a, {}).keys())
    logger.info(f"Tracking {len(all_target_words)} words for {len(target_atoms)} atoms")

    # Collect: atom → [(ctx_text, word, v3_score, a1_resonance, article)]
    atom_contexts: Dict[str, list] = defaultdict(list)
    word_doc_freq: Dict[str, int] = defaultdict(int)

    for _, sentence in sentences:
        tokens = tokenize(sentence)
        tokens_lower = [t.lower() for t in tokens]

        # IDF tracking
        seen = set()
        for t in tokens_lower:
            if t in all_target_words and t not in seen:
                word_doc_freq[t] += 1
                seen.add(t)

        # Context extraction
        for i, tl in enumerate(tokens_lower):
            if tl not in all_target_words:
                continue

            ctx = extract_context(tokens, i)
            if ctx is None:
                continue

            # Which target atoms does this word map to?
            for atom, words in atom_words.items():
                if tl in words:
                    v3 = words[tl]
                    a1 = compute_a1_resonance(tl, atom, word_vectors_48d, atom_centroids_48d)
                    atom_contexts[atom].append((ctx, tl, v3, a1, _))

    for a in target_atoms:
        logger.info(f"  {a}: {len(atom_contexts.get(a, []))} context instances")

    # ----------------------------------------------------------
    # 5. IDF weights
    # ----------------------------------------------------------
    logger.info("=== Step 5: IDF ===")
    idf = {}
    for w, df in word_doc_freq.items():
        idf[w] = np.log(N / df) if df > 0 else 0.0

    # ----------------------------------------------------------
    # 6. Weighted centroids (C_new)
    # ----------------------------------------------------------
    logger.info("=== Step 6: Compute C_new ===")

    C_new = {}
    meta = {}

    for atom in target_atoms:
        contexts = atom_contexts.get(atom, [])
        if not contexts:
            logger.warning(f"  {atom}: no contexts → SKIP")
            meta[atom] = {"support_sentences": 0, "status": "NO_DATA"}
            continue

        # Embed all context texts
        ctx_texts = [c[0] for c in contexts]
        ctx_embs = embedder.embed(ctx_texts)  # (N, 384)

        # Compute weights: M_final = v3^p * a1^q * idf
        weights = []
        word_contrib: Dict[str, float] = defaultdict(float)

        for i, (_, word, v3, a1, _) in enumerate(contexts):
            idf_w = idf.get(word, 1.0)
            if a1 > 0:
                m = (v3 ** P_EXPONENT) * (a1 ** Q_EXPONENT)
            else:
                m = v3 ** P_EXPONENT
            w_final = m * idf_w
            weights.append(w_final)
            word_contrib[word] += w_final

        weights = np.array(weights)

        if weights.sum() <= 0:
            logger.warning(f"  {atom}: zero total weight → SKIP")
            meta[atom] = {"support_sentences": len(contexts), "status": "ZERO_WEIGHT"}
            continue

        # Weighted centroid, normalized
        centroid = (ctx_embs * weights[:, np.newaxis]).sum(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
        C_new[atom] = centroid

        top_words = sorted(word_contrib.items(), key=lambda x: -x[1])[:10]
        top_idx = np.argsort(weights)[-5:][::-1]
        top_examples = [
            {"context": contexts[i][0], "word": contexts[i][1],
             "weight": round(float(weights[i]), 4), "article": contexts[i][4]}
            for i in top_idx
        ]

        meta[atom] = {
            "support_sentences": len(contexts),
            "unique_words": len(word_contrib),
            "total_weight": round(float(weights.sum()), 4),
            "top_contributing_words": [{"word": w, "weight": round(wt, 4)} for w, wt in top_words],
            "top_context_examples": top_examples,
            "status": "OK",
        }
        logger.info(f"  {atom}: {len(contexts)} ctx, {len(word_contrib)} words, top={top_words[0][0]}")

    # ----------------------------------------------------------
    # 7. Dynamic λ blending + drift
    # ----------------------------------------------------------
    logger.info("=== Step 7: Blend + Drift ===")

    C_final = {}
    drift = {}

    for atom in target_atoms:
        if atom not in C_old:
            continue

        old = C_old[atom]

        if atom in C_new:
            new = C_new[atom]
            delta = float(np.dot(old, new) / (np.linalg.norm(old) * np.linalg.norm(new) + 1e-10))

            support = meta[atom]["support_sentences"]
            support_factor = min(1.0, support / SUPPORT_DENOMINATOR)
            drift_factor = 1.0 - delta  # large drift → more update
            lam = support_factor * drift_factor

            blended = (1 - lam) * old + lam * new
            blended = blended / (np.linalg.norm(blended) + 1e-10)
            C_final[atom] = blended

            delta_final = float(np.dot(old, blended) / (np.linalg.norm(old) * np.linalg.norm(blended) + 1e-10))

            drift[atom] = {
                "delta_cosine_old_new": round(delta, 6),
                "delta_cosine_old_final": round(delta_final, 6),
                "support_sentences": support,
                "support_factor": round(support_factor, 4),
                "drift_factor": round(drift_factor, 6),
                "lambda": round(lam, 6),
                "top_contributing_words": meta[atom].get("top_contributing_words", []),
                "top_context_examples": meta[atom].get("top_context_examples", []),
                "status": "BLENDED",
            }
            logger.info(f"  {atom}: δ(old,new)={delta:.4f} λ={lam:.4f} δ(old,final)={delta_final:.4f}")
        else:
            C_final[atom] = old
            drift[atom] = {
                "delta_cosine_old_new": None,
                "delta_cosine_old_final": 1.0,
                "support_sentences": 0,
                "lambda": 0.0,
                "status": "UNCHANGED",
            }
            logger.info(f"  {atom}: UNCHANGED (no data)")

    # ----------------------------------------------------------
    # 8. Save
    # ----------------------------------------------------------
    logger.info("=== Step 8: Save ===")

    atom_ids_out = sorted(C_final.keys())
    Z = np.stack([C_final[a] for a in atom_ids_out])
    np.save(out / "atom_usage_centroids_v1.npy", Z)

    save_meta = {
        "version": "ubaf_v1_prototype",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "embedder": args.embedder,
        "corpus_sentences": N,
        "target_atoms": target_atoms,
        "atom_ids": atom_ids_out,
        "shape": list(Z.shape),
        "params": {
            "context_window": CONTEXT_WINDOW,
            "min_context_tokens": MIN_CONTEXT_TOKENS,
            "p_exponent": P_EXPONENT,
            "q_exponent": Q_EXPONENT,
            "support_denominator": SUPPORT_DENOMINATOR,
        },
    }
    with open(out / "atom_usage_centroids_meta.json", "w") as f:
        json.dump(save_meta, f, indent=2, ensure_ascii=False)

    with open(out / "ubaf_drift_report.json", "w") as f:
        json.dump(drift, f, indent=2, ensure_ascii=False)

    # ----------------------------------------------------------
    # 9. Summary report
    # ----------------------------------------------------------
    logger.info("=== Step 9: Summary ===")
    generate_summary(drift, meta, save_meta, out, time.time() - t0)

    logger.info(f"Done in {time.time() - t0:.1f}s")


def generate_summary(drift, meta, save_meta, out_dir, elapsed):
    """Generate human-readable summary markdown."""
    lines = [
        "# ESDE Phase 8.5 — UBAF Experiment Summary",
        f"Date: {time.strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Parameters",
        f"- Context window: ±{CONTEXT_WINDOW} tokens",
        f"- Weight: v3^{P_EXPONENT} × a1^{Q_EXPONENT} × IDF",
        f"- Support denominator: {SUPPORT_DENOMINATOR}",
        f"- Embedder: {save_meta['embedder']}",
        f"- Corpus: {save_meta['corpus_sentences']} sentences",
        f"- Time: {elapsed:.1f}s",
        "",
        "## Drift Report",
        "",
        "| Atom | δ(old,new) | λ | δ(old,final) | Support | Top Word |",
        "|------|-----------|---|-------------|---------|----------|",
    ]

    for atom in PROTOTYPE_ATOMS:
        d = drift.get(atom, {})
        m = meta.get(atom, {})
        delta_on = d.get("delta_cosine_old_new")
        lam = d.get("lambda", 0)
        delta_of = d.get("delta_cosine_old_final", 1)
        sup = d.get("support_sentences", 0)
        top_w = ""
        tw_list = d.get("top_contributing_words", [])
        if tw_list:
            top_w = tw_list[0]["word"]

        delta_str = f"{delta_on:.4f}" if delta_on is not None else "—"
        lines.append(
            f"| {atom:<15} | {delta_str:>9} | {lam:.4f} | {delta_of:.6f} | {sup:>7} | {top_w} |"
        )

    # Anti-contamination checklist (GPT spec §10)
    lines.extend([
        "",
        "## Anti-Contamination Checklist (GPT Spec §10)",
        "",
    ])
    checks = [
        ("Stopwords excluded?", "YES — STOPWORDS set + min_context filter"),
        ("High-frequency tokens dampened?", "YES — IDF weighting"),
        ("Drift report generated?", "YES — ubaf_drift_report.json"),
        ("Support counts reasonable?", f"See table above (denom={SUPPORT_DENOMINATOR})"),
    ]

    # Check delta < 0.3 for stable atoms
    stable = sum(1 for a, d in drift.items()
                 if d.get("delta_cosine_old_new") is not None
                 and (1 - d["delta_cosine_old_new"]) < 0.3)
    total_blended = sum(1 for d in drift.values() if d.get("status") == "BLENDED")
    checks.append(
        (f"δ < 0.3 drift for stable atoms?",
         f"{stable}/{total_blended} atoms have drift < 0.3")
    )
    checks.append(("Only targeted atoms changed?", f"YES — {len(drift)} prototype atoms only"))

    for label, answer in checks:
        lines.append(f"- **{label}** {answer}")

    lines.extend(["", "---", "*記述せよ、しかし決定するな*"])

    with open(Path(out_dir) / "ubaf_experiment_summary.md", "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Summary → {out_dir}/ubaf_experiment_summary.md")


# ============================================================
# CLI
# ============================================================

def main():
    p = argparse.ArgumentParser(description="ESDE UBAF Centroid Generator")
    p.add_argument("--dictionary", required=True, help="esde_dictionary.json")
    p.add_argument("--synapse", required=True, help="esde_synapses_v3.json")
    p.add_argument("--corpus", required=True, help="Directory with city_*.txt")
    p.add_argument("--a1-final-dir", required=True,
                   help="Directory with *_a1_final.jsonl (integration/lexicon/audit_output/)")
    p.add_argument("--a1-centroids", required=True,
                   help="atom_centroids_48d.csv")
    p.add_argument("--out", default="output/ubaf/", help="Output directory")
    p.add_argument("--embedder", default="minilm", choices=["minilm", "tfidf"])
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
