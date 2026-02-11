#!/usr/bin/env python3
"""
EMO.dislike few-shot experiment via QwQ API.
Run: python3 run_fewshot_dislike.py
"""

import json
import time
import urllib.request

LLM_HOST = "http://100.107.6.119:8001/v1"
LLM_MODEL = "qwq32b_tp2_long32k_existing"
LLM_TIMEOUT = 300
LLM_MAX_TOKENS = 16000
LLM_TEMPERATURE = 0.3

# Load prompts from the markdown file
with open("output/prompt_EMO_dislike_fewshot.md") as f:
    raw = f.read()

# Split on the separator
parts = raw.split("---\n\n# USER PROMPT\n\n")
system_prompt = parts[0].replace("# SYSTEM PROMPT\n\n", "").strip()
user_prompt = parts[1].strip()

print(f"System: {len(system_prompt)} chars")
print(f"User:   {len(user_prompt)} chars")
print(f"Calling QwQ...", flush=True)

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

t0 = time.time()
with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
    data = json.loads(resp.read().decode("utf-8"))
elapsed = time.time() - t0

content = data["choices"][0]["message"]["content"]
print(f"Done in {elapsed:.1f}s ({len(content)} chars)")

# Save raw
with open("output/fewshot_EMO_dislike_raw.txt", "w") as f:
    f.write(content)
print("Saved: output/fewshot_EMO_dislike_raw.txt")
