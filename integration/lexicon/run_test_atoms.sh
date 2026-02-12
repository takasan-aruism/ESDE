#!/bin/bash
# run_test_atoms.sh — COG.know → SOC.public を順番に実行

set -e

echo "=== [1/4] Mapper: COG.know ==="
python3 mapper_a1.py --lexicon-entry lexicon/COG_know.json --dictionary esde_dictionary.json

echo "=== [2/4] Auditor: COG.know ==="
python3 auditor_a1.py --input mapper_output/COG_know_a1.jsonl --dictionary esde_dictionary.json --re-observe

echo "=== [3/4] Mapper: SOC.public ==="
python3 mapper_a1.py --lexicon-entry lexicon/SOC_public.json --dictionary esde_dictionary.json

echo "=== [4/4] Auditor: SOC.public ==="
python3 auditor_a1.py --input mapper_output/SOC_public_a1.jsonl --dictionary esde_dictionary.json --re-observe

echo "=== ALL DONE ==="
