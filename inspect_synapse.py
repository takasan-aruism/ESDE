#!/usr/bin/env python3
"""esde_synapses_v3.json の実際の構造を調査する。"""
import json
from pathlib import Path

data = json.loads(Path("esde_synapses_v3.json").read_text())

print(f"Type: {type(data)}")
print(f"Top-level keys: {list(data.keys()) if isinstance(data, dict) else f'list[{len(data)}]'}")
print()

if isinstance(data, dict):
    for k in list(data.keys())[:5]:
        v = data[k]
        print(f"key={k!r}  type={type(v).__name__}")
        if isinstance(v, dict):
            print(f"  sub-keys: {list(v.keys())[:10]}")
            # Show first edge if exists
            if "edges" in v:
                edges = v["edges"]
                print(f"  edges: {len(edges)} items")
                if edges:
                    print(f"  edges[0] keys: {list(edges[0].keys())}")
                    print(f"  edges[0]: {json.dumps(edges[0], indent=2)[:300]}")
        elif isinstance(v, list):
            print(f"  len={len(v)}")
            if v:
                print(f"  [0] type={type(v[0]).__name__}")
                if isinstance(v[0], dict):
                    print(f"  [0] keys: {list(v[0].keys())[:10]}")
                    print(f"  [0]: {json.dumps(v[0], indent=2)[:300]}")
        print()

# Also grep for "receive" anywhere
print("--- Searching for 'receive.v' in keys/values ---")
found = 0
if isinstance(data, dict):
    for k, v in data.items():
        if "receive" in str(k):
            print(f"  Found in key: {k!r}")
            found += 1
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, list):
                    for item in v2[:3]:
                        if isinstance(item, dict) and "receive" in json.dumps(item):
                            print(f"  Found in {k}/{k2}: {json.dumps(item)[:200]}")
                            found += 1
                            if found > 5: break
                if found > 5: break
        if found > 5: break
if found == 0:
    print("  (not found in first pass — trying flat search)")
    flat = json.dumps(data)
    idx = flat.find("receive")
    if idx >= 0:
        print(f"  Found at char {idx}: ...{flat[max(0,idx-50):idx+100]}...")
