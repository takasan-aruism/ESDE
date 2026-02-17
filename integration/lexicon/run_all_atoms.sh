#!/bin/bash
# ============================================================
# ESDE Lexicon v2 — Full 326-Atom Batch Pipeline
# ============================================================
# Usage:
#   chmod +x run_all_atoms.sh
#   ./run_all_atoms.sh                    # All atoms, single GPU
#
#   # Dual GPU (tp1 × 2) — run in separate terminals:
#   LLM_HOST=http://100.107.6.119:8002/v1 SUBSET=0 ./run_all_atoms.sh
#   LLM_HOST=http://100.107.6.119:8003/v1 SUBSET=1 ./run_all_atoms.sh
#
# Features:
#   - Auto-discovers all lexicon/*.json files
#   - Runs mapper → auditor+re-observe for each atom
#   - Skips already-completed atoms (resume-safe)
#   - SUBSET=0/1 splits atoms for dual-GPU independent runs
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
PARALLEL="${PARALLEL:-4}"  # Concurrent LLM requests per GPU
PARALLEL_REOBS="${PARALLEL_REOBS:-4}"  # Re-observe parallel

# LLM connection (override via env for multi-GPU)
export LLM_HOST="${LLM_HOST:-http://100.107.6.119:8001/v1}"

# Subset: unset=all, 0=even atoms, 1=odd atoms
SUBSET="${SUBSET:-}"

mkdir -p "$MAPPER_OUTPUT" "$AUDIT_OUTPUT" "$LOG_DIR"

# Build atom list (with optional subset filtering)
ALL_FILES=($(ls "$LEXICON_DIR"/*.json 2>/dev/null | sort))
TOTAL_ALL=${#ALL_FILES[@]}

if [ "$TOTAL_ALL" -eq 0 ]; then
    echo "ERROR: No lexicon files found in $LEXICON_DIR/"
    exit 1
fi

if [ -n "$SUBSET" ]; then
    FILTERED=()
    for i in "${!ALL_FILES[@]}"; do
        if [ $((i % 2)) -eq "$SUBSET" ]; then
            FILTERED+=("${ALL_FILES[$i]}")
        fi
    done
    ATOM_FILES=("${FILTERED[@]}")
    SUBSET_LABEL="subset $SUBSET ($([ "$SUBSET" = "0" ] && echo "even" || echo "odd"))"
else
    ATOM_FILES=("${ALL_FILES[@]}")
    SUBSET_LABEL="all"
fi

TOTAL=${#ATOM_FILES[@]}
SUMMARY_FILE="batch_summary$([ -n "$SUBSET" ] && echo "_sub${SUBSET}" || echo "").txt"

echo "============================================================"
echo "  ESDE Full Batch Pipeline"
echo "  Atoms: $TOTAL / $TOTAL_ALL ($SUBSET_LABEL)"
echo "  LLM:   $LLM_HOST"
echo "  Parallel: $PARALLEL (re-observe: $PARALLEL_REOBS)"
echo "  Started: $(date)"
echo "============================================================"

DONE=0
SKIPPED=0
FAILED=0
FAIL_LIST=""
START_TIME=$(date +%s)

for LEXICON_FILE in "${ATOM_FILES[@]}"; do
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

    # Get expected word count from lexicon entry
    EXPECTED_WORDS=$(python3 -c "
import json, sys
e = json.load(open('$LEXICON_FILE'))
words = e.get('core_pool', {}).get('words', [])
print(len(words))
" 2>/dev/null || echo "?")

    # Step 1: Mapper
    echo "  [mapper] $BASENAME ($EXPECTED_WORDS words) ..."

    # Clear mapper output for fresh run
    rm -f "$MAPPER_JSONL"

    # Run mapper in background
    python3 mapper_a1.py \
        --lexicon-entry "$LEXICON_FILE" \
        --dictionary "$DICTIONARY" \
        --parallel "$PARALLEL" \
        >> "$ATOM_LOG" 2>&1 &
    MAPPER_PID=$!

    # Poll progress
    while kill -0 "$MAPPER_PID" 2>/dev/null; do
        if [ -f "$MAPPER_JSONL" ] && [ "$EXPECTED_WORDS" != "?" ] && [ "$EXPECTED_WORDS" -gt 0 ]; then
            CURRENT_LINES=$(wc -l < "$MAPPER_JSONL" 2>/dev/null | tr -d ' ')
            PCT=$((CURRENT_LINES * 100 / EXPECTED_WORDS))
            printf "\r  [mapper] %s %d/%s (%d%%)" "$BASENAME" "$CURRENT_LINES" "$EXPECTED_WORDS" "$PCT"
        fi
        sleep 5
    done
    wait "$MAPPER_PID"
    MAPPER_EXIT=$?
    echo ""

    if [ "$MAPPER_EXIT" -ne 0 ]; then
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
    AUDIT_JSONL="$AUDIT_OUTPUT/${BASENAME}_audit.jsonl"
    rm -f "$AUDIT_JSONL" "$FINAL_JSONL"

    # Run auditor in background
    python3 auditor_a1.py \
        --input "$MAPPER_JSONL" \
        --dictionary "$DICTIONARY" \
        --out-dir "$AUDIT_OUTPUT" \
        --re-observe \
        --parallel "$PARALLEL" \
        --parallel-reobs "$PARALLEL_REOBS" \
        >> "$ATOM_LOG" 2>&1 &
    AUDIT_PID=$!

    # Poll progress (audit JSONL tracks Phase 1, final JSONL appears at completion)
    while kill -0 "$AUDIT_PID" 2>/dev/null; do
        if [ -f "$AUDIT_JSONL" ] && [ "$MAPPER_WORDS" -gt 0 ]; then
            AUDIT_LINES=$(wc -l < "$AUDIT_JSONL" 2>/dev/null | tr -d ' ')
            PCT=$((AUDIT_LINES * 100 / MAPPER_WORDS))
            if [ -f "$FINAL_JSONL" ]; then
                printf "\r  [auditor] %s re-observe in progress..." "$BASENAME"
            else
                printf "\r  [auditor] %s %d/%s (%d%%)" "$BASENAME" "$AUDIT_LINES" "$MAPPER_WORDS" "$PCT"
            fi
        fi
        sleep 5
    done
    wait "$AUDIT_PID"
    AUDIT_EXIT=$?
    echo ""

    if [ "$AUDIT_EXIT" -ne 0 ]; then
        echo "  ❌ AUDITOR FAILED (see $ATOM_LOG)"
        FAILED=$((FAILED + 1))
        FAIL_LIST="$FAIL_LIST $BASENAME(auditor)"
        continue
    fi

    # Quick stats from final
    if [ -f "$FINAL_JSONL" ]; then
        FINAL_WORDS=$(wc -l < "$FINAL_JSONL" | tr -d ' ')
        ATOM_END=$(date +%s)
        ATOM_ELAPSED=$((ATOM_END - ATOM_START))
        if [ "$FINAL_WORDS" -gt 0 ]; then
            SPW=$((ATOM_ELAPSED / FINAL_WORDS))
        else
            SPW=0
        fi
        DONE=$((DONE + 1))
        PROCESSED=$((DONE + FAILED))
        REMAINING=$((TOTAL - DONE - SKIPPED - FAILED))

        # ETA based on running average
        TOTAL_ELAPSED=$((ATOM_END - START_TIME))
        if [ "$PROCESSED" -gt 0 ]; then
            AVG_PER_ATOM=$((TOTAL_ELAPSED / PROCESSED))
            ETA_SEC=$((AVG_PER_ATOM * REMAINING))
            ETA_H=$((ETA_SEC / 3600))
            ETA_M=$(( (ETA_SEC % 3600) / 60 ))
            echo "  ✅ DONE ($FINAL_WORDS words, ${ATOM_ELAPSED}s, ${SPW}s/w) | $DONE done, $REMAINING left, ETA ~${ETA_H}h${ETA_M}m"
        else
            echo "  ✅ DONE ($FINAL_WORDS words, ${ATOM_ELAPSED}s, ${SPW}s/w)"
        fi
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