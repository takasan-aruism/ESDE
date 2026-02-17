#!/usr/bin/env python3
"""Quick test: does Qwen3 return parseable JSON for A1 mapper prompts?"""
import json, os
os.environ.setdefault("LLM_HOST", "http://100.107.6.119:8002/v1")

from mapper_a1 import build_a1_prompt, call_qwq, load_dictionary, get_core_words

dictionary = load_dictionary("esde_dictionary.json")
lexicon = json.load(open("lexicon/ABS_bound.json"))
words = get_core_words(lexicon)

w = words[0]
word = w.get("w", w.get("lemma", ""))
pos = w.get("pos", "n")

atom_id = lexicon.get("atom", lexicon.get("atom_id", ""))
atom_def_entry = dictionary.get(atom_id, {})
atom_def = atom_def_entry.get("definition_en", "")
sym_pair = atom_def_entry.get("symmetric_pair", "")
category = atom_def_entry.get("category", "")

print(f"Testing: {word} ({pos}) for {atom_id}")
prompt = build_a1_prompt(word, pos, atom_id, atom_def, sym_pair, category)

from mapper_a1 import LLM_MODEL, LLM_HOST, LLM_MAX_TOKENS, LLM_TEMPERATURE
print(f"LLM: {LLM_HOST} model={LLM_MODEL}")
print(f"max_tokens={LLM_MAX_TOKENS} temp={LLM_TEMPERATURE}")
print(f"Prompt: {len(prompt)} chars")
print()

text, elapsed = call_qwq(prompt)

print(f"\n=== FULL OUTPUT ({len(text)} chars, {elapsed:.1f}s) ===")
print(text)
print("=== END ===")
