#!/usr/bin/env python3
"""
EMO Category Batch — Generate Lexicon for all 28 remaining EMO atoms.
Uses EMO.like (back 5 axes) as few-shot example.

Usage:
  python3 run_emo_batch.py                    # Run all 28
  python3 run_emo_batch.py EMO.anger          # Run single atom
  python3 run_emo_batch.py EMO.anger EMO.hope # Run specific atoms

Results saved to: emo_batch/results/<atom_id>.json
"""

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

# ── Config ──
LLM_HOST = "http://100.107.6.119:8001/v1"
LLM_MODEL = "qwq32b_tp2_long32k_existing"
LLM_TIMEOUT = 300
LLM_MAX_TOKENS = 16000
LLM_TEMPERATURE = 0.3

SCRIPT_DIR = Path(__file__).parent
DICT_PATH = SCRIPT_DIR / "esde_dictionary.json"
LIKE_PATH = SCRIPT_DIR / "output" / "EMO_like_final.json"
AXES_PATH = SCRIPT_DIR / "definitions" / "axes_levels_v1.json"
RESULTS_DIR = SCRIPT_DIR / "emo_batch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──
with open(DICT_PATH) as f:
    dictionary = json.load(f)
with open(LIKE_PATH) as f:
    like_data = json.load(f)
with open(AXES_PATH) as f:
    axes_def = json.load(f)

# ── Build few-shot example (EMO.like back 5 axes) ──
BACK_AXES = ['resonance', 'symmetry', 'lawfulness', 'experience', 'value_generation']

example_slots = {}
for sk, sd in like_data['slots'].items():
    axis = sk.split('.')[0]
    if axis not in BACK_AXES:
        continue
    if sd.get('na'):
        example_slots[sk] = {"na": True, "na_reason": sd.get("na_reason", "")}
        continue
    clean = [w for w in sd.get('words', []) if not w.get('flags')][:8]
    example_slots[sk] = {
        "na": False,
        "words": [{"w": w["w"], "pos": w["pos"], "reason": w["reason"]} for w in clean]
    }

EXAMPLE_JSON = json.dumps({"atom": "EMO.like", "slots": example_slots}, indent=2, ensure_ascii=False)

# ── Build axes block ──
axes_block = ''
slot_list = []
for axis_id in BACK_AXES:
    ax = axes_def['axes'][axis_id]
    axes_block += f'**{axis_id}** — {ax["name"]} ({ax["definition"]})\n'
    for level_id, level_def in ax['levels'].items():
        axes_block += f'  - {level_id}: {level_def["definition"]}\n'
        slot_list.append(f'{axis_id}.{level_id}')
    axes_block += '\n'

# ── System prompt (shared) ──
SYSTEM_PROMPT = """You are a lexicographer for ESDE (Emergent Semantic Data Engine).

THE CORE PRINCIPLE: "Describe, but do not decide."

You will be given:
1. A COMPLETED EXAMPLE: EMO.like (好) — words assigned to each slot for the back 5 axes
2. A TARGET ATOM: a different EMO atom — you must fill the same slots

RULES:
1. Study the example carefully. Notice HOW words relate to the atom AND the axis/level.
   The example shows the PATTERN — how words sit at the intersection of [atom meaning] × [axis/level meaning].

2. The TARGET ATOM has its own specific meaning. Words must express THAT ATOM specifically.
   Do NOT include generic emotion words. Every word must be specific to the target atom.

3. Do NOT include words that belong to the symmetric pair (opposite atom).

4. Include mixed POS: nouns, verbs, adjectives, adverbs. POS must be one of: n, v, adj, adv.

5. For each word: provide a 1-sentence reason connecting it to BOTH the atom AND the slot.

6. If a slot doesn't apply to this atom, set na:true with a brief reason.

7. Aim for 5-8 words per applicable slot. Quality over quantity.

8. CRITICAL: Avoid "concept labels" that just name the axis.
   Bad: "superficial" in resonance.superficial (that's the axis label)
   Good: "peeve" in resonance.superficial (that's an actual word at surface depth)

9. Output strict JSON only. Same format as the example."""


def build_user_prompt(atom_id: str) -> str:
    """Build user prompt for a specific atom."""
    atom = dictionary['concepts'][atom_id]
    sym_pair = atom.get('symmetric_pair', 'N/A')
    sym_name = dictionary['concepts'].get(sym_pair, {}).get('name', sym_pair)
    
    kanji = atom.get('kanji', '')
    kanji_field = atom.get('kanji_semantic_field', [])
    kanji_str = '\n'.join(f'  - {f}' for f in kanji_field) if kanji_field else '  (not available)'
    
    definition = atom.get('definition_en', atom.get('short_definition', ''))
    triggers = atom.get('triggers_en', [])
    triggers_str = ', '.join(triggers) if triggers else '(not available)'
    
    return f"""### Category
Code: EMO
Name: Emotion

### Target Atom
ID: {atom_id}
Kanji: {kanji}
Kanji semantic field:
{kanji_str}
Definition: {definition}
Key words: {triggers_str}
Symmetric pair (OPPOSITE — do NOT include these words): {sym_pair} ({sym_name})

### Axes and Levels ({len(slot_list)} slots)

{axes_block}

---

### COMPLETED EXAMPLE (EMO.like — study the PATTERN, not the words)

```json
{EXAMPLE_JSON}
```

---

Now generate the same structure for **{atom_id} ({kanji})**.

Remember:
- Words must be specific to {atom_id}, not generic emotions
- Match the PATTERN of the example (atom × axis/level intersection)
- Avoid axis labels as words
- POS: n/v/adj/adv only
- 5-8 words per slot

Output ONLY the JSON."""


def call_qwq(system: str, user: str) -> str:
    """Call QwQ API."""
    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE,
    })
    req = urllib.request.Request(
        f"{LLM_HOST}/chat/completions",
        data=payload.encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def parse_json_response(raw: str) -> dict:
    """Extract JSON from QwQ response (strip thinking tags, markdown).
    Handles common QwQ JSON issues: trailing commas, single quotes,
    comments, control chars, unquoted keys."""
    clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    match = re.search(r'```json\s*\n(.*?)```', clean, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
    else:
        first = clean.find('{')
        last = clean.rfind('}')
        if first >= 0 and last > first:
            extracted = clean[first:last+1]
        else:
            raise ValueError("No JSON found in response")
    
    # Repair pass 1: trailing commas
    extracted = re.sub(r',\s*}', '}', extracted)
    extracted = re.sub(r',\s*]', ']', extracted)
    
    # Try parse
    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        pass
    
    # Repair pass 2: remove // comments
    extracted = re.sub(r'//[^\n]*', '', extracted)
    
    # Repair pass 3: remove control chars (except \n \t)
    extracted = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', extracted)
    
    # Repair pass 4: fix single quotes → double quotes (careful with apostrophes)
    extracted = re.sub(r"(?<=[\[,{:\s])'([^']*?)'(?=[\],}:\s])", r'"\1"', extracted)
    
    # Repair pass 5: fix stray dots/chars before quoted strings
    # e.g. `. "reason"` → `"reason"`
    extracted = re.sub(r',\s*\.\s*"', ', "', extracted)
    
    # Repair pass 6: fix non-ASCII values after "pos": (Chinese char injection)
    # e.g. "pos":得罪 → "pos": "n"  (replace with "n" as safe default)
    extracted = re.sub(r'"pos"\s*:\s*[^\s",}\]]+(?=[^"]*"reason")', '"pos": "n", ', extracted)
    
    # Repair pass 7: fix broken pos values with mixed content
    # e.g. "pos": "v",得罪 → "pos": "v",
    extracted = re.sub(r'("pos"\s*:\s*"[a-z]+"),\s*[^\s"{}\[\]"]+\s*"reason"', r'\1, "reason"', extracted)
    
    # Repair pass 8: trailing commas again after all repairs
    extracted = re.sub(r',\s*}', '}', extracted)
    extracted = re.sub(r',\s*]', ']', extracted)
    
    # Repair pass 6: truncated response — find last complete slot
    try:
        return json.loads(extracted)
    except json.JSONDecodeError as e:
        # Try to salvage by finding the last valid closing brace pair
        # Walk backwards to find a point where JSON is valid
        for i in range(len(extracted) - 1, max(0, len(extracted) - 2000), -1):
            if extracted[i] == '}':
                # Try closing at this point with enough braces
                attempt = extracted[:i+1]
                # Count open/close braces
                opens = attempt.count('{') - attempt.count('}')
                attempt += '}' * opens
                try:
                    return json.loads(attempt)
                except json.JSONDecodeError:
                    continue
        
        # Last resort: raise original error with context
        pos = e.pos
        context = extracted[max(0,pos-80):pos+80]
        raise ValueError(f"JSON repair failed at pos {pos}: {e.msg}\nContext: ...{context}...")


def quick_stats(data: dict) -> dict:
    """Quick quality stats."""
    total = 0
    na_count = 0
    pos_counts = {}
    for sk, sd in data.get('slots', {}).items():
        if sd.get('na'):
            na_count += 1
            continue
        for w in sd.get('words', []):
            total += 1
            pos = w.get('pos', '?')
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
    return {
        'total_words': total,
        'na_slots': na_count,
        'active_slots': len(data.get('slots', {})) - na_count,
        'pos': pos_counts,
        'avg_per_slot': round(total / max(1, len(data.get('slots', {})) - na_count), 1),
    }


def process_atom(atom_id: str) -> dict:
    """Generate lexicon for one atom."""
    result_path = RESULTS_DIR / f"{atom_id.replace('.', '_')}.json"
    
    print(f"\n{'='*60}")
    print(f"  {atom_id}")
    print(f"{'='*60}")
    
    # Build prompt
    user_prompt = build_user_prompt(atom_id)
    print(f"  Prompt: {len(SYSTEM_PROMPT) + len(user_prompt):,} chars")
    
    # Call QwQ (with retry)
    MAX_RETRIES = 3
    parsed = None
    raw = ""
    elapsed = 0
    
    try:
        for attempt in range(1, MAX_RETRIES + 1):
            if attempt > 1:
                print(f"  Retry {attempt}/{MAX_RETRIES}...", end='', flush=True)
            else:
                print(f"  Calling QwQ...", end='', flush=True)
            
            t0 = time.time()
            raw = call_qwq(SYSTEM_PROMPT, user_prompt)
            elapsed = time.time() - t0
            print(f" {elapsed:.1f}s ({len(raw):,} chars)")
            
            try:
                parsed = parse_json_response(raw)
                break
            except (ValueError, json.JSONDecodeError) as e:
                print(f"  ⚠️ Parse failed (attempt {attempt}): {e}")
                if attempt == MAX_RETRIES:
                    raise
    except Exception as e:
        print(f"  ❌ FAILED after {MAX_RETRIES} attempts: {e}")
        # Save raw for debugging
        with open(result_path.with_suffix('.raw.txt'), 'w') as f:
            f.write(raw)
        print(f"  Raw saved: {result_path.with_suffix('.raw.txt').name}")
        return {'atom': atom_id, 'status': 'error', 'error': str(e)}
    
    try:
        stats = quick_stats(parsed)
        print(f"  Words: {stats['total_words']}  Active slots: {stats['active_slots']}  N/A: {stats['na_slots']}")
        print(f"  POS: {stats['pos']}  Avg/slot: {stats['avg_per_slot']}")
        
        # Show sample words per slot
        for sk, sd in parsed.get('slots', {}).items():
            if sd.get('na'):
                continue
            words = [w['w'] for w in sd.get('words', [])[:4]]
            print(f"    {sk:35s} {', '.join(words)}")
        
        # Save
        output = {
            'atom': atom_id,
            'source': 'qwq_fewshot_from_EMO.like',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'elapsed_s': round(elapsed, 1),
            'stats': stats,
            'data': parsed,
            'raw_length': len(raw),
        }
        with open(result_path, 'w') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved: {result_path.name}")
        
        return {'atom': atom_id, 'status': 'ok', 'stats': stats}
    
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        # Save raw for debugging
        with open(result_path.with_suffix('.raw.txt'), 'w') as f:
            f.write(raw)
        return {'atom': atom_id, 'status': 'error', 'error': str(e)}


def main():
    # Determine which atoms to process
    emo_atoms = {k: v for k, v in dictionary['concepts'].items() 
                 if v.get('category') == 'EMO' and k not in ('EMO.like',)}
    
    if len(sys.argv) > 1:
        targets = sys.argv[1:]
    else:
        targets = sorted(emo_atoms.keys())
    
    print(f"EMO Batch: {len(targets)} atoms to process")
    print(f"Few-shot example: EMO.like (back 5 axes)")
    print(f"Output: {RESULTS_DIR}/")
    
    results = []
    for atom_id in targets:
        if atom_id not in dictionary['concepts']:
            print(f"  ⚠️ {atom_id} not found in dictionary, skipping")
            continue
        r = process_atom(atom_id)
        results.append(r)
    
    # Summary
    ok = sum(1 for r in results if r['status'] == 'ok')
    err = sum(1 for r in results if r['status'] == 'error')
    print(f"\n{'='*60}")
    print(f"  BATCH COMPLETE: {ok} ok, {err} errors out of {len(results)}")
    print(f"{'='*60}")
    
    # Save summary
    with open(RESULTS_DIR / '_summary.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
