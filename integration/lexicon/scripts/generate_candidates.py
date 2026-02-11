#!/usr/bin/env python3
"""
Phase B Step B1: QwQ Candidate Generation for Lexicon
Generates word candidates for all 48 slots of a given Atom.

Usage:
    python generate_candidates.py \
        --atom EMO.like \
        --definitions-dir ../definitions \
        --output ../output/qwq_candidates_EMO.like.json

    # Dry run (print prompt only):
    python generate_candidates.py \
        --atom EMO.like \
        --definitions-dir ../definitions \
        --output /dev/null \
        --dry-run
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

# QwQ Connection
LLM_HOST = "http://100.107.6.119:8001/v1"
LLM_MODEL = "qwq32b_tp2_long32k_existing"
LLM_TIMEOUT = 180
LLM_MAX_TOKENS = 16000
LLM_TEMPERATURE = 0.3

# ── System Prompt ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are assisting with the construction of a Lexicon for ESDE
(Emergent Semantic Data Engine), a semantic coordinate system rooted in Aruism philosophy.

THE CORE PRINCIPLE OF ARUISM: "Describe, but do not decide."
You are an observer mapping a territory, not a judge choosing answers.
Your task is NOT to find the "correct" word for each slot.
Your task IS to describe what kinds of words would naturally land in each region
of the coordinate space when encountered in real English text.

IMAGINE THIS: Someone reads a news article, a novel, a conversation transcript.
They encounter a word. That word, given its meaning and usage, would naturally
"fall into" a specific coordinate. You are listing words that fall into each slot.

RULES:
1. Atoms are POSITIONS (coordinates), not meanings. You are mapping words to positions.
2. Propose English words that a reader might encounter in real text.
   Include nouns, verbs, adjectives, and adverbs — a good slot has mixed POS.
3. For each word, specify POS as one of: noun, verb, adj, adv.
4. For each word, give a 1-sentence reason explaining WHY it lands in this coordinate.
5. If a slot genuinely does not apply to this Atom in direct/literal usage, say so.
   Writing "N/A" is a valid and respected observation — it is better to mark N/A
   than to force abstract or metaphorical words into a slot where they don't belong.
6. Do NOT propose words that would better belong to the symmetric pair.
7. Stick to direct/literal usage. Metaphorical, poetic, or domain-specific usage is out of scope.
8. Aim for 5-15 words per applicable slot. More is better than fewer.
   If you can only think of 1-3, that is also fine — quality over quantity.
9. AVOID "concept labels" — words like "harmony", "unity", "essence", "balance"
   that describe the axis definition itself rather than words that would appear
   in natural text and land at this coordinate. Ask yourself:
   "Would I encounter this word in a newspaper article about someone's feelings?"
10. The `kanji_semantic_field` (English) is the authoritative definition.
    The kanji character is a supplementary conceptual anchor, not a definition source.
    If the kanji suggests meanings outside the semantic field, ignore them.

Output format (strict JSON):
{
  "atom": "EMO.like",
  "slots": {
    "temporal.emergence": {
      "words": [
        {"word": "infatuation", "pos": "noun", "reason": "sudden onset of passionate liking — encountered in text describing new romance"},
        {"word": "smitten", "pos": "adj", "reason": "describes the state of being suddenly struck with liking"},
        {"word": "fancy", "pos": "verb", "reason": "to suddenly take a liking to someone/something"},
        ...
      ],
      "na": false
    },
    "scale.stellar": {
      "words": [],
      "na": true,
      "na_reason": "Emotion of liking has no direct/literal application at planetary/stellar scale"
    },
    ...all 48 slots...
  }
}

Think step by step. Output FINAL: followed by the JSON only."""


# ── Prompt Builder ─────────────────────────────────────────────

def build_user_prompt(atom_def: dict, atom_id: str, axes_levels: dict, category_def: dict) -> str:
    """Build user prompt with full 4-layer context (English only)."""

    # Layer 1: Category
    cat_block = f"""### Category
Code: {atom_def['category']}
Name: {category_def['name']}
Description: {category_def['description']}"""

    # Layer 2: Atom + Kanji
    kanji_field = "\n".join(f"  - {f}" for f in atom_def['kanji_semantic_field'])
    atom_block = f"""### Atom
ID: {atom_id}
Kanji: {atom_def['kanji']}
Kanji semantic field (authoritative English definitions derived from the kanji):
{kanji_field}
Definition: {atom_def['short_definition']}

Symmetric pair: {atom_def['symmetric_pair']}
WARNING: Do NOT propose words that belong to {atom_def['symmetric_pair']} (the opposite coordinate).
If a word could belong to either side, skip it."""

    # Layer 3-4: All axes and levels
    axes_block = "### Axes and Levels (all 10 axes, 48 levels total)\n\n"
    slot_list = []
    for axis_id, axis_def in axes_levels['axes'].items():
        axes_block += f"**{axis_id}** — {axis_def['name']} ({axis_def['definition']})\n"
        for level_id, level_def in axis_def['levels'].items():
            axes_block += f"  - {level_id}: {level_def['definition']}\n"
            slot_list.append(f"{axis_id}.{level_id}")
        axes_block += "\n"

    # Guidance block
    guidance = """### Guidance

Remember: "Describe, but do not decide."

For each slot, ask yourself:
- "If I were reading a real English text and encountered a word expressing EMO.like,
   AND that word's usage specifically aligned with this axis+level combination,
   what word might that be?"
- "Would this word actually appear in a newspaper, novel, or conversation?"
- "Is this word specific to EMO.like, or is it just a generic label for the axis?"

If the answer to the last question is "it's just a label for the axis", do NOT include it.
If no real-world words fit a slot in direct/literal usage, mark it N/A.
N/A is an observation, not a failure.

Aim for mixed POS (nouns, verbs, adjectives, adverbs) in each slot."""

    # Slots to fill
    slots_block = "### Slots to fill:\n\n"
    for slot in slot_list:
        slots_block += f"{slot}:\n"

    return f"""{cat_block}

{atom_block}

{axes_block}

{guidance}

{slots_block}

Provide words for EACH of the {len(slot_list)} slots above.
Mark inapplicable slots as na: true with a reason.
Include a mix of nouns, verbs, adjectives, and adverbs where possible.""", slot_list


# ── QwQ Caller ─────────────────────────────────────────────────

def call_qwq(system_prompt: str, user_prompt: str) -> str:
    """Call QwQ via OpenAI-compatible API."""
    import urllib.request

    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE,
    })

    req = urllib.request.Request(
        f"{LLM_HOST}/chat/completions",
        data=payload.encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    print(f"  Calling QwQ ({LLM_MODEL})... ", end="", flush=True)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0
    print(f"done ({elapsed:.1f}s)")

    return data["choices"][0]["message"]["content"]


# ── Response Parser ────────────────────────────────────────────

def parse_qwq_response(response: str) -> dict:
    """Extract JSON from QwQ response (after FINAL: marker)."""
    # QwQ outputs reasoning then FINAL:
    for marker in ["FINAL:", "Final:", "final:"]:
        if marker in response:
            parts = response.split(marker, 1)
            if len(parts) > 1:
                response = parts[1].strip()
                break

    # Remove markdown code fences
    response = re.sub(r'```json\s*', '', response)
    response = re.sub(r'```\s*$', '', response)
    response = response.strip()

    return json.loads(response)


# ── POS Normalization ──────────────────────────────────────────

POS_MAP = {
    "noun": "n", "n": "n",
    "verb": "v", "v": "v",
    "adj": "adj", "adjective": "adj", "a": "adj", "s": "adj",
    "adv": "adv", "adverb": "adv", "r": "adv",
}

def normalize_pos(pos: str) -> str:
    """Normalize POS to canonical form: n/v/adj/adv."""
    return POS_MAP.get(pos.lower().strip(), pos.lower().strip())


# ── Normalization ──────────────────────────────────────────────

def normalize_word(w: str) -> str:
    """Normalize word: lowercase, trim, underscore→space, collapse spaces."""
    w = w.strip().lower()
    w = w.replace("_", " ")
    w = re.sub(r'\s+', ' ', w)
    return w


# ── Post-Processing ───────────────────────────────────────────

def postprocess_candidates(raw: dict, slot_list: list) -> dict:
    """
    Normalize QwQ output to canonical form:
    - words as {w, pos, reason, status}
    - POS normalized to n/v/adj/adv
    - Word lemmas normalized
    - Dedup within each slot
    """
    result = {
        "atom": raw["atom"],
        "slots": {},
    }

    for slot_key in slot_list:
        slot_data = raw.get("slots", {}).get(slot_key, {})

        if slot_data.get("na", False):
            result["slots"][slot_key] = {
                "words": [],
                "na": True,
                "na_reason": slot_data.get("na_reason", "Not specified"),
            }
            continue

        words_raw = slot_data.get("words", [])
        words_normalized = []
        seen = set()

        for entry in words_raw:
            if isinstance(entry, str):
                w = normalize_word(entry)
                pos = "n"  # default if QwQ forgot POS
            elif isinstance(entry, dict):
                w = normalize_word(entry.get("word", entry.get("w", "")))
                pos = normalize_pos(entry.get("pos", "n"))
            else:
                continue

            if not w:
                continue

            key = f"{w}::{pos}"
            if key in seen:
                continue
            seen.add(key)

            reason = ""
            if isinstance(entry, dict):
                reason = entry.get("reason", "")

            words_normalized.append({
                "w": w,
                "pos": pos,
                "reason": reason,
                "status": "proposed",  # All QwQ candidates start as proposed
            })

        result["slots"][slot_key] = {
            "words": words_normalized,
            "na": False,
        }

    return result


# ── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate Lexicon candidates via QwQ")
    parser.add_argument("--atom", required=True, help="Atom ID (e.g. EMO.like)")
    parser.add_argument("--definitions-dir", required=True, help="Path to definitions/")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt only, don't call QwQ")
    args = parser.parse_args()

    defs_dir = Path(args.definitions_dir)

    # Load definitions
    print("Loading definitions...")
    with open(defs_dir / "categories_v1.json") as f:
        categories = json.load(f)
    with open(defs_dir / "axes_levels_v1.json") as f:
        axes_levels = json.load(f)
    with open(defs_dir / "atoms_v1.json") as f:
        atoms = json.load(f)

    atom_id = args.atom
    if atom_id not in atoms["atoms"]:
        print(f"ERROR: Atom '{atom_id}' not found in atoms_v1.json")
        sys.exit(1)

    atom_def = atoms["atoms"][atom_id]
    category_def = categories["categories"][atom_def["category"]]

    # Build prompt
    print(f"Building prompt for {atom_id}...")
    user_prompt, slot_list = build_user_prompt(atom_def, atom_id, axes_levels, category_def)

    print(f"  Slots: {len(slot_list)}")
    print(f"  System prompt: {len(SYSTEM_PROMPT)} chars")
    print(f"  User prompt: {len(user_prompt)} chars")

    if args.dry_run:
        print("\n=== SYSTEM PROMPT ===")
        print(SYSTEM_PROMPT)
        print("\n=== USER PROMPT ===")
        print(user_prompt)
        print("\n[dry-run] No QwQ call made.")
        return

    # Call QwQ
    response_text = call_qwq(SYSTEM_PROMPT, user_prompt)

    # Save raw response
    raw_path = Path(args.output).with_suffix(".raw.txt")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w") as f:
        f.write(response_text)
    print(f"  Raw response saved: {raw_path}")

    # Parse
    print("  Parsing response...")
    try:
        parsed = parse_qwq_response(response_text)
    except json.JSONDecodeError as e:
        print(f"  ERROR: Failed to parse JSON: {e}")
        print(f"  Raw response saved to {raw_path} for manual inspection.")
        sys.exit(1)

    # Post-process
    print("  Post-processing...")
    candidates = postprocess_candidates(parsed, slot_list)

    # Stats
    total_words = 0
    na_slots = 0
    pos_counts = {"n": 0, "v": 0, "adj": 0, "adv": 0}
    for slot_key, slot_data in candidates["slots"].items():
        if slot_data.get("na"):
            na_slots += 1
        else:
            for w in slot_data["words"]:
                total_words += 1
                pos_counts[w["pos"]] = pos_counts.get(w["pos"], 0) + 1

    active = len(slot_list) - na_slots
    print(f"  Results: {total_words} words across {active} active slots ({na_slots} N/A)")
    print(f"  POS: noun={pos_counts.get('n',0)} verb={pos_counts.get('v',0)} adj={pos_counts.get('adj',0)} adv={pos_counts.get('adv',0)}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()