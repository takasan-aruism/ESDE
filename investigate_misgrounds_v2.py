#!/usr/bin/env python3
"""
investigate_misgrounds_v2.py — 逆方向探索
==========================================
「EMO.like に繋がる全 synset」を列挙し、receive との関係を調べる。

Part A: 逆引き — target atom への edge を持つ全 synset
Part B: SynapseGrounder の実際の経路を再現
Part C: パッチファイル全走査
"""

import json
from pathlib import Path

SYNAPSE_PATH = "esde_synapses_v3.json"
PATCH_PATHS = list(Path("patches").glob("*.json")) if Path("patches").exists() else []

REVERSE_TARGETS = {
    "EMO.like":    "receive",
    "ACT.give":    "use",
    "SOC.request": "offer",
    "EMO.pride":   "win",
    "COM.answer":  "cover",
}


def main():
    raw = json.loads(Path(SYNAPSE_PATH).read_text())
    synapses = raw.get("synapses", raw)
    print(f"Base synapse: {len(synapses)} synsets\n")

    # ═══ Part A: 逆引き ═══
    print("=" * 70)
    print("Part A: 逆引き — target atom への edge を持つ全 synset")
    print("=" * 70)

    for target_atom, expected_verb in REVERSE_TARGETS.items():
        print(f"\n### {target_atom} (misground verb: {expected_verb})")
        print(f"| synset_id | raw_score | weight | rank | lemma-in-synset |")
        print(f"|-----------|-----------|--------|------|-----------------|")
        found = 0
        for sid, edges in synapses.items():
            if sid.startswith("_"):
                continue
            if not isinstance(edges, list):
                continue
            for edge in edges:
                if edge.get("concept_id") == target_atom:
                    found += 1
                    # Check if expected_verb appears as lemma in this synset
                    has_verb = ""
                    try:
                        from nltk.corpus import wordnet as wn
                        s = wn.synset(sid)
                        lemma_names = [l.name() for l in s.lemmas()]
                        if expected_verb in lemma_names:
                            has_verb = f" ← **{expected_verb} IS A LEMMA**"
                        else:
                            has_verb = f" lemmas: {lemma_names[:5]}"
                    except Exception:
                        has_verb = ""
                    
                    score = edge.get("raw_score", "?")
                    score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
                    weight = edge.get("weight", "?")
                    weight_str = f"{weight:.4f}" if isinstance(weight, (int, float)) else str(weight)
                    print(f"| `{sid}` | {score_str} | {weight_str} | {edge.get('rank','?')} | {has_verb} |")
        if found == 0:
            print(f"| (none found) | — | — | — | — |")
        print(f"  → Total: {found} synsets point to {target_atom}")

    # ═══ Part B: SynapseGrounder 経路再現 ═══
    print("\n" + "=" * 70)
    print("Part B: WordNet synset 展開 — 各 verb lemma の全 synset（POS=VERB）")
    print("        synapse に存在するかどうかをチェック")
    print("=" * 70)

    try:
        from nltk.corpus import wordnet as wn
        wordnet_ok = True
    except Exception:
        wordnet_ok = False
        print("(WordNet not available — skipping Part B)")

    if wordnet_ok:
        for target_atom, verb in REVERSE_TARGETS.items():
            wn_synsets = wn.synsets(verb, pos=wn.VERB)
            print(f"\n### {verb} (target: {target_atom})")
            print(f"  WordNet verb synsets: {len(wn_synsets)}")
            for s in wn_synsets:
                sid = s.name()
                in_synapse = sid in synapses
                edges_info = ""
                if in_synapse:
                    edges = synapses[sid]
                    atoms = [f"{e.get('concept_id','?')}" for e in edges]
                    has_target = "★ HAS TARGET" if target_atom in atoms else ""
                    edges_info = f"  edges → {', '.join(atoms)} {has_target}"
                else:
                    edges_info = "  (NOT in Synapse)"
                print(f"  - `{sid}`: {s.definition()[:60]}")
                print(f"    in_synapse={in_synapse}{edges_info}")

    # ═══ Part C: パッチファイル走査 ═══
    print("\n" + "=" * 70)
    print("Part C: パッチファイル全走査")
    print("=" * 70)

    for patch_path in sorted(PATCH_PATHS):
        print(f"\n### {patch_path}")
        try:
            patch_data = json.loads(patch_path.read_text())
            if isinstance(patch_data, list):
                entries = patch_data
            elif isinstance(patch_data, dict) and "entries" in patch_data:
                entries = patch_data["entries"]
            else:
                entries = [patch_data]
            
            for target_atom, verb in REVERSE_TARGETS.items():
                hits = [e for e in entries 
                        if isinstance(e, dict) and (
                            e.get("atom", "") == target_atom or
                            e.get("concept_id", "") == target_atom or
                            verb in e.get("synset_id", "") or
                            verb in e.get("edge_key", "")
                        )]
                if hits:
                    print(f"  {verb}/{target_atom}: {len(hits)} entries found")
                    for h in hits:
                        print(f"    {json.dumps(h)[:200]}")
        except Exception as ex:
            print(f"  Error: {ex}")

    if not PATCH_PATHS:
        print("  (no patch files found in patches/)")

    # ═══ Part D: relation_logger の grounding 経路を確認 ═══
    print("\n" + "=" * 70)
    print("Part D: SynapseGrounder ソースコード確認")
    print("=" * 70)
    
    grounder_path = Path("integration/relations/relation_logger.py")
    if grounder_path.exists():
        src = grounder_path.read_text()
        # Find the ground_verb method
        import re
        # Look for how synsets are looked up and how atoms are selected
        for pattern in [r"def ground_verb.*?(?=\n    def |\nclass |\Z)", 
                        r"synsets.*?=.*?\n",
                        r"get_edges.*?\n"]:
            matches = re.findall(pattern, src, re.DOTALL)
            for m in matches[:2]:
                lines = m.strip().split("\n")[:30]
                print("  " + "\n  ".join(lines))
                print("  ...")
    else:
        print(f"  {grounder_path} not found")


if __name__ == "__main__":
    main()
