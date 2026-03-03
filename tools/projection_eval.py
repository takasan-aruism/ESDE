#!/usr/bin/env python3
"""
ESDE Projection Operator — Evaluation Scorer
=============================================
Computes Recall@1, Recall@3 for projection operator experiments.

Usage:
  python3 tools/projection_eval.py \
    --pred-jsonl output/projection_eval/BC/pred_50.jsonl \
    --gt-jsonl output/projection_eval/ground_truth_50.jsonl

Ground Truth schema (JSONL):
{
  "id": "berlin_0001",
  "sentence": "Berlin is the capital and largest city of Germany.",
  "targets": [
    {
      "span_text": "capital",
      "span": [14, 21],       // nullable
      "pos": "NOUN",
      "atoms_top3": ["SOC.official", "SPC.place", "SOC.city"]
    }
  ],
  "notes": "winner=null; acceptable set is top3"
}

Prediction schema (JSONL):
{
  "id": "berlin_0001",
  "targets": [
    {
      "span_text": "capital",
      "pred_top3": ["SOC.official", "SPC.place", "STA.wealth"],
      "scores_top3": [0.41, 0.33, 0.08]
    }
  ]
}

Metric:
  hit@k = 1 if any(gt_atom in pred_topk[:k])
  Recall@k = mean(hit@k) over all targets

Author: Claude (Implementation) per GPT spec §1.2
Date: 2026-03-03
"""

import json
import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def load_jsonl(path: str) -> List[dict]:
    """Load JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def match_targets(gt_targets: List[dict], pred_targets: List[dict]) -> List[Tuple[dict, dict]]:
    """
    Match GT targets to prediction targets by span_text.
    Returns list of (gt_target, pred_target) pairs.
    Unmatched GT targets get pred_target=None.
    """
    pred_by_text = {}
    for pt in pred_targets:
        key = pt.get("span_text", "").lower().strip()
        pred_by_text[key] = pt
    
    pairs = []
    for gt in gt_targets:
        key = gt.get("span_text", "").lower().strip()
        pred = pred_by_text.get(key)
        pairs.append((gt, pred))
    
    return pairs


def compute_hit(gt_atoms: List[str], pred_atoms: List[str], k: int) -> int:
    """
    hit@k = 1 if any gt_atom appears in pred_atoms[:k].
    """
    if not gt_atoms or not pred_atoms:
        return 0
    pred_set = set(pred_atoms[:k])
    for ga in gt_atoms:
        if ga in pred_set:
            return 1
    return 0


def evaluate(gt_path: str, pred_path: str) -> dict:
    """
    Main evaluation.
    
    Returns:
        {
            "recall_at_1": float,
            "recall_at_3": float,
            "n_targets": int,
            "n_matched": int,
            "n_unmatched": int,
            "per_target": [...],
            "confusions": [...]
        }
    """
    gt_records = load_jsonl(gt_path)
    pred_records = load_jsonl(pred_path)
    
    # Index predictions by id
    pred_by_id = {r["id"]: r for r in pred_records}
    
    hits_at_1 = []
    hits_at_3 = []
    per_target = []
    confusions = []
    n_matched = 0
    n_unmatched = 0
    
    for gt_rec in gt_records:
        rec_id = gt_rec["id"]
        pred_rec = pred_by_id.get(rec_id)
        
        if pred_rec is None:
            # No prediction for this sentence
            for gt_t in gt_rec.get("targets", []):
                n_unmatched += 1
                hits_at_1.append(0)
                hits_at_3.append(0)
                per_target.append({
                    "id": rec_id,
                    "span_text": gt_t.get("span_text", ""),
                    "gt_atoms": gt_t.get("atoms_top3", []),
                    "pred_top3": [],
                    "hit_at_1": 0,
                    "hit_at_3": 0,
                    "status": "MISSING_PRED",
                })
            continue
        
        pairs = match_targets(
            gt_rec.get("targets", []),
            pred_rec.get("targets", [])
        )
        
        for gt_t, pred_t in pairs:
            gt_atoms = gt_t.get("atoms_top3", [])
            span_text = gt_t.get("span_text", "")
            
            if pred_t is None:
                n_unmatched += 1
                pred_top3 = []
                h1, h3 = 0, 0
                status = "UNMATCHED"
            else:
                n_matched += 1
                pred_top3 = pred_t.get("pred_top3", [])
                h1 = compute_hit(gt_atoms, pred_top3, 1)
                h3 = compute_hit(gt_atoms, pred_top3, 3)
                status = "HIT" if h3 else "MISS"
            
            hits_at_1.append(h1)
            hits_at_3.append(h3)
            
            per_target.append({
                "id": rec_id,
                "span_text": span_text,
                "gt_atoms": gt_atoms,
                "pred_top3": pred_top3,
                "hit_at_1": h1,
                "hit_at_3": h3,
                "status": status,
            })
            
            # Confusion: where pred_top1 missed
            if h1 == 0 and pred_top3:
                confusions.append({
                    "id": rec_id,
                    "span_text": span_text,
                    "gt_atoms": gt_atoms,
                    "pred_top1": pred_top3[0] if pred_top3 else "",
                    "pred_top3": pred_top3,
                })
    
    n_targets = len(hits_at_1)
    recall_at_1 = sum(hits_at_1) / max(n_targets, 1)
    recall_at_3 = sum(hits_at_3) / max(n_targets, 1)
    
    return {
        "recall_at_1": recall_at_1,
        "recall_at_3": recall_at_3,
        "n_targets": n_targets,
        "n_matched": n_matched,
        "n_unmatched": n_unmatched,
        "per_target": per_target,
        "confusions": confusions,
    }


def print_report(result: dict, mode: str = ""):
    """Print evaluation report to stdout."""
    header = f"=== Projection Eval: {mode} ===" if mode else "=== Projection Eval ==="
    print(header)
    print(f"Targets: {result['n_targets']}  (matched={result['n_matched']}, unmatched={result['n_unmatched']})")
    print(f"Recall@1: {result['recall_at_1']:.3f}")
    print(f"Recall@3: {result['recall_at_3']:.3f}")
    print()
    
    # Per-target detail
    print(f"{'ID':<16} {'Word':<16} {'Hit@3':<6} {'GT atoms':<40} {'Pred top3'}")
    print("-" * 110)
    for t in result["per_target"]:
        gt_str = ", ".join(t["gt_atoms"][:3])
        pred_str = ", ".join(t["pred_top3"][:3])
        hit_marker = "✅" if t["hit_at_3"] else "❌"
        print(f"{t['id']:<16} {t['span_text']:<16} {hit_marker:<6} {gt_str:<40} {pred_str}")
    
    # Confusions
    if result["confusions"]:
        print()
        print("--- Confusions (pred_top1 missed GT set) ---")
        for c in result["confusions"][:20]:
            print(f"  {c['span_text']:<16} pred={c['pred_top1']:<20} gt={c['gt_atoms']}")


def export_confusions_csv(confusions: list, path: str):
    """Export confusion table as CSV."""
    if not confusions:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "span_text", "gt_atoms", "pred_top1", "pred_top3"])
        writer.writeheader()
        for c in confusions:
            row = dict(c)
            row["gt_atoms"] = "|".join(row["gt_atoms"])
            row["pred_top3"] = "|".join(row["pred_top3"])
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="ESDE Projection Eval Scorer")
    parser.add_argument("--pred-jsonl", required=True, help="Predictions JSONL file")
    parser.add_argument("--gt-jsonl", required=True, help="Ground truth JSONL file")
    parser.add_argument("--mode", default="", help="Mode label (for reporting)")
    parser.add_argument("--out-csv", default=None, help="Export confusion CSV")
    parser.add_argument("--out-json", default=None, help="Export full results JSON")
    args = parser.parse_args()
    
    result = evaluate(args.gt_jsonl, args.pred_jsonl)
    print_report(result, mode=args.mode)
    
    if args.out_csv:
        export_confusions_csv(result["confusions"], args.out_csv)
        print(f"\nConfusions exported to: {args.out_csv}")
    
    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Full results exported to: {args.out_json}")


if __name__ == "__main__":
    main()
