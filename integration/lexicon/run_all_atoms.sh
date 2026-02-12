#!/bin/bash
# ============================================================
# ESDE Lexicon v2 — Full 326-Atom Batch Pipeline
# ============================================================
# Usage:
#   chmod +x run_all_atoms.sh
#   ./run_all_atoms.sh
#
# Features:
#   - Auto-discovers all lexicon/*.json files
#   - Runs mapper → auditor+re-observe for each atom
#   - Skips already-completed atoms (resume-safe)
#   - Logs per-atom timing and results
#   - Generates summary report at the end
#
# To re-run a specific atom, delete its *_a1_final.jsonl:
#   rm audit_output/EMO_like_a1_final.jsonl
#   ./run_all_atoms.sh
# ============================================================

set -o pipefail

LEXICON_DIR="lexicon"
DICTIONARY="esde_dictionary.json"
MAPPER_OUTPUT="mapper_output"
AUDIT_OUTPUT="audit_output"
LOG_DIR="logs"
SUMMARY_FILE="batch_summary.txt"

mkdir -p "$MAPPER_OUTPUT" "$AUDIT_OUTPUT" "$LOG_DIR"

# Count atoms
TOTAL=$(ls "$LEXICON_DIR"/*.json 2>/dev/null | wc -l | tr -d ' ')
if [ "$TOTAL" -eq 0 ]; then
    echo "ERROR: No lexicon files found in $LEXICON_DIR/"
    exit 1
fi

echo "============================================================"
echo "  ESDE Full Batch Pipeline"
echo "  Atoms: $TOTAL"
echo "  Started: $(date)"
echo "============================================================"

DONE=0
SKIPPED=0
FAILED=0
FAIL_LIST=""
START_TIME=$(date +%s)

for LEXICON_FILE in "$LEXICON_DIR"/*.json; do
    # Extract atom ID from filename: lexicon/EMO_like.json → EMO_like
    BASENAME=$(basename "$LEXICON_FILE" .json)
    # Convert to JSONL name: EMO_like → EMO_like_a1
    MAPPER_JSONL="$MAPPER_OUTPUT/${BASENAME}_a1.jsonl"
    FINAL_JSONL="$AUDIT_OUTPUT/${BASENAME}_a1_final.jsonl"
    ATOM_LOG="$LOG_DIR/${BASENAME}.log"

    CURRENT=$((DONE + SKIPPED + FAILED + 1))

    # Skip if already completed
    if [ -f "$FINAL_JSONL" ]; then
        SKIPPED=$((SKIPPED + 1))
        echo "[$CURRENT/$TOTAL] $BASENAME → SKIP (already done)"
        continue
    fi

    echo ""
    echo "[$CURRENT/$TOTAL] $BASENAME"
    ATOM_START=$(date +%s)

    # Step 1: Mapper
    echo "  [mapper] $BASENAME ..."
    if ! python3 mapper_a1.py \
        --lexicon-entry "$LEXICON_FILE" \
        --dictionary "$DICTIONARY" \
        >> "$ATOM_LOG" 2>&1; then
        echo "  ❌ MAPPER FAILED (see $ATOM_LOG)"
        FAILED=$((FAILED + 1))
        FAIL_LIST="$FAIL_LIST $BASENAME(mapper)"
        continue
    fi

    # Check mapper output exists
    if [ ! -f "$MAPPER_JSONL" ]; then
        echo "  ❌ MAPPER OUTPUT MISSING: $MAPPER_JSONL"
        FAILED=$((FAILED + 1))
        FAIL_LIST="$FAIL_LIST $BASENAME(no_output)"
        continue
    fi

    MAPPER_WORDS=$(wc -l < "$MAPPER_JSONL" | tr -d ' ')

    # Step 2: Auditor + Re-observe
    echo "  [auditor] $BASENAME ($MAPPER_WORDS words) ..."
    if ! python3 auditor_a1.py \
        --input "$MAPPER_JSONL" \
        --dictionary "$DICTIONARY" \
        --out-dir "$AUDIT_OUTPUT" \
        --re-observe \
        >> "$ATOM_LOG" 2>&1; then
        echo "  ❌ AUDITOR FAILED (see $ATOM_LOG)"
        FAILED=$((FAILED + 1))
        FAIL_LIST="$FAIL_LIST $BASENAME(auditor)"
        continue
    fi

    ATOM_END=$(date +%s)
    ATOM_ELAPSED=$((ATOM_END - ATOM_START))

    # Quick stats from final
    if [ -f "$FINAL_JSONL" ]; then
        FINAL_WORDS=$(wc -l < "$FINAL_JSONL" | tr -d ' ')
        echo "  ✅ DONE ($FINAL_WORDS words, ${ATOM_ELAPSED}s)"
        DONE=$((DONE + 1))
    else
        echo "  ❌ FINAL OUTPUT MISSING"
        FAILED=$((FAILED + 1))
        FAIL_LIST="$FAIL_LIST $BASENAME(no_final)"
    fi
done

END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))
HOURS=$((TOTAL_ELAPSED / 3600))
MINS=$(( (TOTAL_ELAPSED % 3600) / 60 ))

echo ""
echo "============================================================"
echo "  BATCH COMPLETE"
echo "  Done: $DONE  Skipped: $SKIPPED  Failed: $FAILED  Total: $TOTAL"
echo "  Time: ${HOURS}h ${MINS}m (${TOTAL_ELAPSED}s)"
echo "  Finished: $(date)"
if [ -n "$FAIL_LIST" ]; then
    echo "  Failed atoms:$FAIL_LIST"
fi
echo "============================================================"

# Write summary
cat > "$SUMMARY_FILE" << EOF
ESDE Batch Pipeline Summary
============================
Date: $(date)
Total atoms: $TOTAL
Done: $DONE
Skipped: $SKIPPED
Failed: $FAILED
Total time: ${HOURS}h ${MINS}m (${TOTAL_ELAPSED}s)
Failed atoms:$FAIL_LIST
EOF

echo "Summary written to $SUMMARY_FILE"
echo ""
echo "Next step: run python3 batch_report.py to generate full statistics"
