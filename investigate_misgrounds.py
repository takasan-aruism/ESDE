#!/usr/bin/env python3
"""
investigate_misgrounds.py — ESDE Misground Edge Investigation
=============================================================
Block-1 Step 1: base synapse から CONSISTENT_MISGROUND 対象の edge_key を特定する。

Usage (リポジトリルートで):
    python investigate_misgrounds.py

Actual base synapse JSON structure:
    {
      "_meta": {...},
      "synapses": {
        "synset_id": [
          {"concept_id": "XXX.yyy", "raw_score": 0.xx, "axis": "...",
           "lemma": "...", "pos": "v", "weight": ..., "rank": ...},
          ...
        ]
      }
    }
"""

import json
from pathlib import Path

SYNAPSE_PATH = "esde_synapses_v3.json"

TARGETS = [
    ("receive", "EMO.like",    1, '"受け取る"と"好む"は意味が異なる'),
    ("use",     "ACT.give",    1, '"使う"と"与える"は逆方向'),
    ("offer",   "SOC.request", 1, '"提供する"と"要求する"は真逆'),
    ("win",     "EMO.pride",   2, '"勝つ"は事象、"誇り"は感情。因果≠等価'),
    ("cover",   "COM.answer",  2, '"覆う/報道する"と"答える"。文脈依存が強すぎる'),
]


def get_gloss(synset_id: str) -> str:
    try:
        from nltk.corpus import wordnet as wn
        return wn.synset(synset_id).definition()
    except Exception:
        return "—"


def fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)


def main():
    raw = json.loads(Path(SYNAPSE_PATH).read_text())
    synapses = raw.get("synapses", raw)
    print(f"Loaded: {len(synapses)} synsets\n")

    results = []
    patch_p1 = []

    for priority in (1, 2):
        label = "Priority 1: 初手 tombstone 候補" if priority == 1 else "Priority 2: 次巡候補"
        print(f"## {label}\n")
        print("| lemma | synset_id | concept_id | raw_score | weight | rank | edge_key | gloss |")
        print("|-------|-----------|------------|-----------|--------|------|----------|-------|")

        for lemma, wrong_atom, pri, reason in TARGETS:
            if pri != priority:
                continue

            prefix = f"{lemma}.v."
            found = []

            for sid, edges in synapses.items():
                if sid.startswith("_"):
                    continue
                if not sid.startswith(prefix):
                    continue
                if not isinstance(edges, list):
                    continue
                for edge in edges:
                    if edge.get("concept_id") == wrong_atom:
                        gloss = get_gloss(sid)
                        entry = {
                            "synset_id": sid,
                            "concept_id": edge["concept_id"],
                            "raw_score": edge.get("raw_score", "?"),
                            "weight": edge.get("weight", "?"),
                            "rank": edge.get("rank", "?"),
                            "axis": edge.get("axis", "?"),
                            "level": edge.get("level", "?"),
                            "edge_key": f"{sid}::{edge['concept_id']}",
                            "gloss": gloss,
                        }
                        found.append(entry)
                        g = gloss[:55] if len(gloss) > 55 else gloss
                        print(f"| {lemma} | `{sid}` | {edge['concept_id']} | {fmt(edge.get('raw_score','?'))} | {fmt(edge.get('weight','?'))} | {edge.get('rank','?')} | `{entry['edge_key']}` | {g} |")

            if not found:
                print(f"| {lemma} | **NOT FOUND** | {wrong_atom} | — | — | — | — | — |")

            results.append({
                "lemma": lemma, "wrong_atom": wrong_atom,
                "priority": priority, "reason": reason,
                "found": len(found) > 0, "edges": found,
            })
            if priority == 1:
                patch_p1.extend(found)

        print()

    # ── 補足: 各 lemma の全 verb synset + 全 edge 一覧 ──
    print("## 補足: 各 lemma の全 verb synset 一覧\n")
    for lemma, wrong_atom, _, _ in TARGETS:
        prefix = f"{lemma}.v."
        sids = sorted(sid for sid in synapses if sid.startswith(prefix))
        print(f"### {lemma}: {len(sids)} verb synsets in base  (target misground: {wrong_atom})")
        for sid in sids:
            edges = synapses[sid]
            gloss = get_gloss(sid)
            print(f"  - `{sid}`: {gloss[:80]}")
            for e in edges:
                marker = " ← **MISGROUND TARGET**" if e.get("concept_id") == wrong_atom else ""
                print(f"    [{e.get('rank','?')}] {e.get('concept_id','?')}  score={fmt(e.get('raw_score','?'))}  weight={fmt(e.get('weight','?'))}{marker}")
        print()

    # ── tombstone パッチ候補 ──
    print("## tombstone パッチ候補（Priority 1 のみ）\n")
    print("```json")
    tombstones = []
    for e in patch_p1:
        t = {
            "op": "disable_edge",
            "edge_key": e["edge_key"],
            "synset_id": e["synset_id"],
            "atom": e["concept_id"],
            "score": 0.0,
            "reason": "misground_hotfix_v3.3",
        }
        tombstones.append(t)
        print(json.dumps(t, ensure_ascii=False))
    print("```")
    print(f"\n合計 {len(tombstones)} 件の disable_edge 候補")

    out = Path("investigate_misgrounds_result.json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n機械可読結果 → {out}")


if __name__ == "__main__":
    main()
