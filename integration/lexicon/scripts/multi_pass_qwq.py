#!/usr/bin/env python3
"""
Lexicon Multi-Pass QwQ Pipeline
================================
Applies the 3AI workflow (Design → Audit → Implementation) within QwQ.

Pass A: Design    — Describe what each slot means for this atom (no words yet)
Pass B: Audit     — Check the design descriptions for accuracy
Pass C: Implement — Generate actual word candidates using the audited design
Pass D: Audit     — Check the generated words for quality issues
Pass E: Re-impl   — Fix words based on audit feedback
Pass F: Finalize  — Extract clean JSON

Usage:
    # Full pipeline:
    python multi_pass_qwq.py --atom EMO.like --definitions-dir ../definitions --output-dir ../output

    # Single pass (for debugging):
    python multi_pass_qwq.py --atom EMO.like --definitions-dir ../definitions --output-dir ../output --pass A

    # Resume from a specific pass (uses saved intermediate files):
    python multi_pass_qwq.py --atom EMO.like --definitions-dir ../definitions --output-dir ../output --resume-from C
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

# ── QwQ Connection ─────────────────────────────────────────────
LLM_HOST = "http://100.107.6.119:8001/v1"
LLM_MODEL = "qwq32b_tp2_long32k_existing"
LLM_TIMEOUT = 300   # longer timeout for multi-context passes
LLM_MAX_TOKENS = 16000
LLM_TEMPERATURE = 0.3
LLM_TEMPERATURE_AUDIT = 0.2  # lower for audit passes

# ── Shared Context Builder ─────────────────────────────────────

def build_atom_context(atom_def: dict, atom_id: str, axes_levels: dict, category_def: dict) -> str:
    """Build the shared context block used across all passes."""
    kanji_field = '\n'.join(f'  - {f}' for f in atom_def['kanji_semantic_field'])

    axes_block = ''
    slot_list = []
    for axis_id, axis_def in axes_levels['axes'].items():
        axes_block += f'**{axis_id}** — {axis_def["name"]} ({axis_def["definition"]})\n'
        for level_id, level_def in axis_def['levels'].items():
            axes_block += f'  - {level_id}: {level_def["definition"]}\n'
            slot_list.append(f'{axis_id}.{level_id}')
        axes_block += '\n'

    context = f"""### Category
Code: {atom_def['category']}
Name: {category_def['name']}
Description: {category_def['description']}

### Atom
ID: {atom_id}
Kanji: {atom_def['kanji']}
Kanji semantic field (authoritative):
{kanji_field}
Definition: {atom_def['short_definition']}
Symmetric pair: {atom_def['symmetric_pair']}

### Axes and Levels ({len(slot_list)} slots)

{axes_block}"""

    return context, slot_list


# ═══════════════════════════════════════════════════════════════
# PASS A: DESIGN — Describe each slot's meaning for this atom
# ═══════════════════════════════════════════════════════════════

PASS_A_SYSTEM = """You are a semantic analyst for ESDE (Emergent Semantic Data Engine).

THE CORE PRINCIPLE: "Describe, but do not decide."

Your task is to DESCRIBE what each coordinate slot means for a specific atom.
You are NOT generating words yet. You are writing a brief guide that explains:
- What this axis+level combination means when applied to this atom
- What kind of real-world situations or expressions would fall here
- Whether this slot has direct/literal applicability (or should be N/A)

Think concretely. Use examples from everyday life, not abstract definitions.
Bad: "temporal.emergence means the first appearance of favorable regard"
Good: "temporal.emergence for EMO.like: the moment someone first feels attracted — 
       like when you meet someone at a party and feel an instant connection,
       or try a new food and immediately love it"

For slots that don't apply in direct/literal usage, mark them N/A and explain why.
N/A is a valid and respected observation."""

PASS_A_USER_TEMPLATE = """{context}

---

For EACH of the {n_slots} slots listed above, write:
1. A 1-2 sentence description of what this slot means for {atom_id}
2. A concrete everyday example of when/where you'd encounter this
3. Whether it's APPLICABLE or N/A (with reason if N/A)

Format:
```
SLOT: temporal.emergence
DESCRIPTION: The moment favorable regard first appears — a sudden feeling of liking.
EXAMPLE: Meeting someone at a party and feeling instant attraction; trying a new dish and immediately loving it.
STATUS: APPLICABLE

SLOT: scale.stellar
DESCRIPTION: N/A
EXAMPLE: N/A
STATUS: N/A — Emotional liking does not operate at planetary/stellar scale in literal usage.
```

Go through all {n_slots} slots. Be concrete, not abstract."""


# ═══════════════════════════════════════════════════════════════
# PASS B: AUDIT DESIGN — Check descriptions for accuracy
# ═══════════════════════════════════════════════════════════════

PASS_B_SYSTEM = """You are a design auditor for ESDE (Emergent Semantic Data Engine).

You are reviewing a DESIGN DOCUMENT that describes what each coordinate slot means 
for a specific atom. Your job is to check for:

1. VAGUENESS — Is the description concrete enough to guide word selection?
   Flag descriptions that just restate the axis definition without adding atom-specific meaning.
2. N/A ACCURACY — Should this slot really be N/A? Or was the designer too conservative/too liberal?
3. EXAMPLE QUALITY — Does the example actually illustrate the slot, or is it generic?
4. SLOT CONFUSION — Is the description actually describing a DIFFERENT slot?
   e.g., describing "emergence" content under "establishment"
5. SYMMETRIC PAIR LEAK — Does the description drift toward the opposite atom?

For each slot, output: PASS, REVISE (with specific fix), or FLAG (needs human review).
Be strict but fair. A good design enables good implementation."""

PASS_B_USER_TEMPLATE = """{context}

---

Here is the DESIGN DOCUMENT to audit:

{pass_a_output}

---

Audit each slot. Output format:
```
SLOT: temporal.emergence
VERDICT: PASS
NOTE: (optional clarification)

SLOT: resonance.structural  
VERDICT: REVISE
ISSUE: Description is too abstract — restates axis definition instead of connecting to EMO.like
FIX: Should describe liking based on shared structure, e.g., "compatible personalities" or "like-minded"

SLOT: scale.stellar
VERDICT: PASS
NOTE: N/A decision is correct
```

Go through ALL slots."""


# ═══════════════════════════════════════════════════════════════
# PASS C: IMPLEMENT — Generate word candidates using audited design
# ═══════════════════════════════════════════════════════════════

PASS_C_SYSTEM = """You are a lexicographer for ESDE (Emergent Semantic Data Engine).

THE CORE PRINCIPLE: "Describe, but do not decide."

You have been given a DESIGN GUIDE that describes what each slot means for this atom.
Your task is to LIST WORDS that would naturally land in each slot when encountered 
in real English text.

RULES:
1. Use the design guide as your map — it tells you what to look for in each slot.
2. Propose words a reader would encounter in real text (newspapers, novels, conversations).
3. Include mixed POS: nouns, verbs, adjectives, adverbs.
4. For each word: specify POS (noun/verb/adj/adv) and a 1-sentence reason.
5. N/A slots from the design: keep as N/A.
6. Do NOT propose words belonging to the symmetric pair ({symmetric_pair}).
7. Stick to direct/literal usage. No metaphor, no domain jargon.
8. Aim for 5-10 words per applicable slot.
9. CRITICAL: Avoid "concept labels" that just name the axis.
   Ask: "Would I encounter this word in a news article about someone's feelings?"
   If the answer is no, don't include it.

Output strict JSON:
{{
  "atom": "{atom_id}",
  "slots": {{
    "temporal.emergence": {{
      "words": [
        {{"word": "infatuation", "pos": "noun", "reason": "sudden onset of passionate liking"}},
        ...
      ],
      "na": false
    }},
    "scale.stellar": {{
      "words": [],
      "na": true,
      "na_reason": "reason from design guide"
    }}
  }}
}}"""

PASS_C_USER_TEMPLATE = """{context}

---

### Design Guide (audited)

{audited_design}

---

Generate word candidates for ALL slots. Output ONLY the JSON."""


# ═══════════════════════════════════════════════════════════════
# PASS D: AUDIT IMPLEMENTATION — Check generated words
# ═══════════════════════════════════════════════════════════════

PASS_D_SYSTEM = """You are an implementation auditor for ESDE (Emergent Semantic Data Engine).

You are reviewing WORD CANDIDATES generated for a lexicon. Check each word against:

1. AXIS LABEL TEST — Is this word just naming the axis/level itself?
   e.g., "emergent" in lawfulness.emergent, "essential" in resonance.essential
   → REJECT: this is a concept label, not an EMO.like word

2. ATOM SPECIFICITY TEST — Is this word specific to this atom, or could it go in ANY atom?
   e.g., "transform" could go in any atom's symmetry.transformative slot
   → REJECT: not atom-specific

3. POS ACCURACY — Is the claimed POS correct?
   e.g., "smitten" claimed as noun but it's actually adj
   → FIX: correct the POS

4. SYMMETRIC PAIR TEST — Does this word belong to the opposite atom?
   e.g., "aversion" proposed for EMO.like → belongs to EMO.dislike
   → REJECT

5. NATURAL TEXT TEST — Would you actually encounter this word in natural English text?
   e.g., "primordial" in value_generation.sacred for EMO.like → too abstract
   → REJECT or REPLACE

For each word: ACCEPT, REJECT (with reason), or REPLACE (with suggested replacement).
Also note if a slot needs MORE words (target: 5-10 per applicable slot).

Be strict. Quality over quantity."""

PASS_D_USER_TEMPLATE = """{context}

---

### Generated Candidates to Audit

{pass_c_output}

---

Audit EVERY word in EVERY slot. Output format:
```
SLOT: temporal.emergence
  infatuation (noun) → ACCEPT
  smitten (adj) → ACCEPT
  drawn (adj) → ACCEPT
  COVERAGE: 3 words — needs more (target 5-10)
  SUGGESTIONS: crush(n), captivate(v), beguile(v)

SLOT: resonance.structural
  patterned (adj) → REJECT — axis label, not EMO.like specific
  systematic (adj) → REJECT — axis label
  organized (adj) → REJECT — axis label
  COVERAGE: 0 accepted — slot needs complete rework
  SUGGESTIONS: compatible(adj), like-minded(adj), simpatico(adj)

SLOT: scale.stellar
  N/A → ACCEPT
```

Go through ALL slots."""


# ═══════════════════════════════════════════════════════════════
# PASS E: RE-IMPLEMENT — Fix based on audit
# ═══════════════════════════════════════════════════════════════

PASS_E_SYSTEM = """You are a lexicographer for ESDE (Emergent Semantic Data Engine).

You are given:
1. The original generated candidates
2. An audit report with ACCEPT/REJECT/REPLACE decisions

Your task: produce a CORRECTED version.
- Keep all ACCEPTED words
- Remove all REJECTED words  
- Apply all REPLACE suggestions
- Add suggested words where coverage was flagged as insufficient
- Ensure 5-10 words per applicable slot where possible

Output strict JSON only (same format as the original candidates).
Do NOT include any explanation or reasoning — just the corrected JSON."""

PASS_E_USER_TEMPLATE = """### Original Candidates

{pass_c_output}

---

### Audit Report

{pass_d_output}

---

Produce the corrected JSON. Keep accepted words, remove rejected, apply replacements and additions.
Output ONLY the JSON."""


# ═══════════════════════════════════════════════════════════════
# PASS F: FINALIZE — Clean JSON extraction
# ═══════════════════════════════════════════════════════════════
# Pass F is not an LLM call — it's a Python post-processing step
# that normalizes the output from Pass E.


# ── QwQ Caller ─────────────────────────────────────────────────

def call_qwq(system_prompt: str, user_prompt: str, temperature: float = None,
             enable_thinking: bool = True) -> str:
    """Call QwQ via OpenAI-compatible API."""
    import urllib.request

    if temperature is None:
        temperature = LLM_TEMPERATURE

    # For passes that don't need CoT, we can try to suppress thinking
    # by asking for direct output (QwQ may still think internally)
    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": temperature,
    })

    req = urllib.request.Request(
        f"{LLM_HOST}/chat/completions",
        data=payload.encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0

    content = data["choices"][0]["message"]["content"]
    return content, elapsed


# ── JSON Parser (robust) ──────────────────────────────────────

def extract_json(text: str) -> dict:
    """Extract JSON from QwQ output, handling various formats."""
    # Strategy 1: ```json fence
    match = re.search(r'```json\s*\n(.*?)```', text, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
    else:
        # Strategy 2: first { to last }
        first = text.find('{')
        last = text.rfind('}')
        if first != -1 and last > first:
            extracted = text[first:last + 1]
        else:
            raise ValueError("No JSON found in response")

    # Repair common QwQ typos
    extracted = re.sub(r':\.\s*"', ': "', extracted)
    extracted = re.sub(r',\s*}', '}', extracted)
    extracted = re.sub(r',\s*]', ']', extracted)

    return json.loads(extracted)


# ── Thinking Removal ──────────────────────────────────────────

def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from QwQ output."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


# ── Pass Runner ────────────────────────────────────────────────

def run_pass(pass_name: str, system_prompt: str, user_prompt: str,
             output_dir: Path, atom_id: str, temperature: float = None) -> str:
    """Run a single pass, save raw output, return content."""
    safe_atom = atom_id.replace('.', '_')
    raw_path = output_dir / f"pass_{pass_name}_{safe_atom}.raw.txt"
    
    print(f"\n{'='*60}")
    print(f"  PASS {pass_name}")
    print(f"{'='*60}")
    print(f"  System prompt: {len(system_prompt)} chars")
    print(f"  User prompt:   {len(user_prompt)} chars")
    print(f"  Calling QwQ... ", end="", flush=True)

    content, elapsed = call_qwq(system_prompt, user_prompt, temperature)

    print(f"done ({elapsed:.1f}s)")
    print(f"  Response: {len(content)} chars")

    # Save raw
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, 'w') as f:
        f.write(content)
    print(f"  Saved: {raw_path}")

    return content


# ── Normalization (Pass F) ─────────────────────────────────────

POS_MAP = {
    "noun": "n", "n": "n",
    "verb": "v", "v": "v",
    "adj": "adj", "adjective": "adj", "a": "adj", "s": "adj",
    "adv": "adv", "adverb": "adv", "r": "adv",
}

def normalize_final(raw_json: dict, slot_list: list) -> dict:
    """Normalize to canonical output format."""
    result = {"atom": raw_json.get("atom", ""), "slots": {}}

    for slot_key in slot_list:
        sd = raw_json.get("slots", {}).get(slot_key, {})

        if sd.get("na", False):
            result["slots"][slot_key] = {
                "words": [],
                "na": True,
                "na_reason": sd.get("na_reason", "Not specified"),
            }
            continue

        words = []
        seen = set()
        for entry in sd.get("words", []):
            if isinstance(entry, dict):
                w = entry.get("word", entry.get("w", "")).strip().lower()
                pos = POS_MAP.get(entry.get("pos", "noun").lower().strip(), "n")
                reason = entry.get("reason", "")
            else:
                continue

            if not w:
                continue
            key = f"{w}::{pos}"
            if key in seen:
                continue
            seen.add(key)

            words.append({
                "w": w, "pos": pos,
                "reason": reason, "status": "proposed",
            })

        result["slots"][slot_key] = {"words": words, "na": False}

    return result


# ── Main Pipeline ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-pass QwQ Lexicon Pipeline")
    parser.add_argument("--atom", required=True)
    parser.add_argument("--definitions-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pass", dest="single_pass", default=None,
                        help="Run only this pass (A/B/C/D/E/F)")
    parser.add_argument("--resume-from", default=None,
                        help="Resume from this pass, loading earlier results from files")
    args = parser.parse_args()

    defs_dir = Path(args.definitions_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_atom = args.atom.replace('.', '_')

    # Load definitions
    print("Loading definitions...")
    with open(defs_dir / "categories_v1.json") as f:
        categories = json.load(f)
    with open(defs_dir / "axes_levels_v1.json") as f:
        axes_levels = json.load(f)
    with open(defs_dir / "atoms_v1.json") as f:
        atoms = json.load(f)

    atom_id = args.atom
    atom_def = atoms["atoms"][atom_id]
    cat_def = categories["categories"][atom_def["category"]]

    context, slot_list = build_atom_context(atom_def, atom_id, axes_levels, cat_def)

    # Determine which passes to run
    all_passes = ['A', 'B', 'C', 'D', 'E', 'F']
    if args.single_pass:
        passes_to_run = [args.single_pass.upper()]
    elif args.resume_from:
        start_idx = all_passes.index(args.resume_from.upper())
        passes_to_run = all_passes[start_idx:]
    else:
        passes_to_run = all_passes

    print(f"Atom: {atom_id}")
    print(f"Passes to run: {' → '.join(passes_to_run)}")

    # Helper to load previous pass output from file
    def load_pass(pass_name: str) -> str:
        path = output_dir / f"pass_{pass_name}_{safe_atom}.raw.txt"
        if not path.exists():
            print(f"  ERROR: {path} not found. Run pass {pass_name} first.")
            sys.exit(1)
        with open(path) as f:
            return f.read()

    # ── PASS A: Design ──
    pass_a_output = None
    if 'A' in passes_to_run:
        user_prompt = PASS_A_USER_TEMPLATE.format(
            context=context, n_slots=len(slot_list), atom_id=atom_id)
        pass_a_output = run_pass('A', PASS_A_SYSTEM, user_prompt, output_dir, atom_id)
    elif any(p in passes_to_run for p in ['B', 'C', 'D', 'E', 'F']):
        pass_a_output = load_pass('A')
        print(f"  Loaded Pass A from file ({len(pass_a_output)} chars)")

    # ── PASS B: Audit Design ──
    pass_b_output = None
    if 'B' in passes_to_run:
        user_prompt = PASS_B_USER_TEMPLATE.format(
            context=context, pass_a_output=strip_thinking(pass_a_output))
        pass_b_output = run_pass('B', PASS_B_SYSTEM, user_prompt, output_dir, atom_id,
                                 temperature=LLM_TEMPERATURE_AUDIT)
    elif any(p in passes_to_run for p in ['C', 'D', 'E', 'F']):
        pass_b_output = load_pass('B')
        print(f"  Loaded Pass B from file ({len(pass_b_output)} chars)")

    # ── Build Audited Design (merge A + B for Pass C) ──
    audited_design = None
    if pass_a_output and pass_b_output:
        # Combine: original design + audit notes
        audited_design = f"""## Original Design

{strip_thinking(pass_a_output)}

## Audit Notes

{strip_thinking(pass_b_output)}"""

    # ── PASS C: Implement ──
    pass_c_output = None
    if 'C' in passes_to_run:
        system_prompt = PASS_C_SYSTEM.format(
            atom_id=atom_id, symmetric_pair=atom_def['symmetric_pair'])
        user_prompt = PASS_C_USER_TEMPLATE.format(
            context=context, audited_design=audited_design or strip_thinking(pass_a_output))
        pass_c_output = run_pass('C', system_prompt, user_prompt, output_dir, atom_id)
    elif any(p in passes_to_run for p in ['D', 'E', 'F']):
        pass_c_output = load_pass('C')
        print(f"  Loaded Pass C from file ({len(pass_c_output)} chars)")

    # ── PASS D: Audit Implementation ──
    pass_d_output = None
    if 'D' in passes_to_run:
        user_prompt = PASS_D_USER_TEMPLATE.format(
            context=context, pass_c_output=strip_thinking(pass_c_output))
        pass_d_output = run_pass('D', PASS_D_SYSTEM, user_prompt, output_dir, atom_id,
                                 temperature=LLM_TEMPERATURE_AUDIT)
    elif any(p in passes_to_run for p in ['E', 'F']):
        pass_d_output = load_pass('D')
        print(f"  Loaded Pass D from file ({len(pass_d_output)} chars)")

    # ── PASS E: Re-implement ──
    pass_e_output = None
    if 'E' in passes_to_run:
        user_prompt = PASS_E_USER_TEMPLATE.format(
            pass_c_output=strip_thinking(pass_c_output),
            pass_d_output=strip_thinking(pass_d_output))
        pass_e_output = run_pass('E', PASS_E_SYSTEM, user_prompt, output_dir, atom_id)
    elif 'F' in passes_to_run:
        pass_e_output = load_pass('E')
        print(f"  Loaded Pass E from file ({len(pass_e_output)} chars)")

    # ── PASS F: Finalize (local processing, no LLM) ──
    if 'F' in passes_to_run:
        print(f"\n{'='*60}")
        print(f"  PASS F (local)")
        print(f"{'='*60}")

        try:
            raw_json = extract_json(strip_thinking(pass_e_output))
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  ERROR parsing Pass E output: {e}")
            print(f"  Check pass_E_{safe_atom}.raw.txt manually")
            sys.exit(1)

        final = normalize_final(raw_json, slot_list)

        # Stats
        total = 0
        na_count = 0
        pos_counts = {}
        for sk, sd in final["slots"].items():
            if sd.get("na"):
                na_count += 1
            else:
                for w in sd["words"]:
                    total += 1
                    pos_counts[w["pos"]] = pos_counts.get(w["pos"], 0) + 1

        active = len(slot_list) - na_count
        print(f"  Total: {total} words across {active} active slots ({na_count} N/A)")
        print(f"  POS: {pos_counts}")

        # Save final
        final_path = output_dir / f"lexicon_{safe_atom}_final.json"
        with open(final_path, 'w') as f:
            json.dump(final, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {final_path}")

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
