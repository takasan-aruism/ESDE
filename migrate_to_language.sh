#!/bin/bash
# ESDE Consolidation — Migration Script
# Date: 2026-03-20
# Purpose: Create language/ structure under ESDE-Research and organize original ESDE
#
# Run from ~/esde/
# IMPORTANT: Review the plan before executing. Run with --dry-run first.
#
# Usage:
#   cd ~/esde
#   bash migrate_to_language.sh --dry-run    # Preview only
#   bash migrate_to_language.sh              # Execute

set -euo pipefail

ESDE_ROOT="$(cd "$(dirname "$0")" && pwd)"
RESEARCH_ROOT="$ESDE_ROOT/Research"
LANG_ROOT="$RESEARCH_ROOT/language"
LEGACY_ROOT="$ESDE_ROOT/legacy/20260320_consolidation"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE — no files will be moved ==="
fi

do_cmd() {
    if $DRY_RUN; then
        echo "  [DRY] $*"
    else
        eval "$@"
    fi
}

echo ""
echo "============================================================"
echo "ESDE Consolidation Migration"
echo "============================================================"
echo "ESDE root:     $ESDE_ROOT"
echo "Research root:  $RESEARCH_ROOT"
echo "Language root:  $LANG_ROOT"
echo "Legacy target:  $LEGACY_ROOT"
echo ""

# ============================================================
# Step 1: Create language/ directory structure
# ============================================================
echo "--- Step 1: Create language/ directory structure ---"

DIRS=(
    "$LANG_ROOT"
    "$LANG_ROOT/atoms"
    "$LANG_ROOT/atoms/a1_batch"
    "$LANG_ROOT/synapse"
    "$LANG_ROOT/synapse/patches"
    "$LANG_ROOT/lexicon"
    "$LANG_ROOT/lexicon/data"
    "$LANG_ROOT/lexicon/data/lexicon_entries"
    "$LANG_ROOT/lexicon/data/expanded"
    "$LANG_ROOT/lexicon/data/definitions"
    "$LANG_ROOT/lexicon/data/mapper_output"
    "$LANG_ROOT/projection"
    "$LANG_ROOT/sensor"
    "$LANG_ROOT/relations"
    "$LANG_ROOT/tests"
    "$LANG_ROOT/docs"
    "$LANG_ROOT/docs/experiment_reports"
)

for d in "${DIRS[@]}"; do
    do_cmd "mkdir -p '$d'"
done

echo ""

# ============================================================
# Step 2: Copy Active Core files → language/
# ============================================================
echo "--- Step 2: Copy Active Core → language/ ---"

# --- Atoms ---
echo "  [atoms] Dictionary + A1 batch"
do_cmd "cp '$ESDE_ROOT/integration/lexicon/esde_dictionary.json' '$LANG_ROOT/atoms/' 2>/dev/null || true"
do_cmd "cp '$ESDE_ROOT/esde_dictionary.json' '$LANG_ROOT/atoms/' 2>/dev/null || true"
# A1 batch files (326 atom JSONL files)
do_cmd "cp '$ESDE_ROOT/integration/lexicon/lexicon/'*.jsonl '$LANG_ROOT/atoms/a1_batch/' 2>/dev/null || true"
do_cmd "cp '$ESDE_ROOT/integration/lexicon/lexicon/'*.json '$LANG_ROOT/atoms/a1_batch/' 2>/dev/null || true"

# --- Synapse ---
echo "  [synapse] Base + patches + store"
do_cmd "cp '$ESDE_ROOT/esde_synapses_v3.json' '$LANG_ROOT/synapse/'"
do_cmd "cp '$ESDE_ROOT/synapse/'*.py '$LANG_ROOT/synapse/'"
do_cmd "cp '$ESDE_ROOT/patches/'* '$LANG_ROOT/synapse/patches/'"
do_cmd "cp '$ESDE_ROOT/synapse_profiles.json' '$LANG_ROOT/synapse/' 2>/dev/null || true"

# --- Lexicon pipeline code ---
echo "  [lexicon] Pipeline scripts"
do_cmd "cp '$ESDE_ROOT/integration/lexicon/wn_'*.py '$LANG_ROOT/lexicon/'"
do_cmd "cp '$ESDE_ROOT/integration/lexicon/auditor_a1.py' '$LANG_ROOT/lexicon/'"
do_cmd "cp '$ESDE_ROOT/integration/lexicon/mapper_a1.py' '$LANG_ROOT/lexicon/'"
do_cmd "cp '$ESDE_ROOT/integration/lexicon/batch_report.py' '$LANG_ROOT/lexicon/'"
do_cmd "cp '$ESDE_ROOT/integration/lexicon/synapse_v4_'*.py '$LANG_ROOT/lexicon/'"

# Lexicon active data
echo "  [lexicon] Active data (entries, expanded, definitions, mapper)"
do_cmd "cp -r '$ESDE_ROOT/integration/lexicon/lexicon/'* '$LANG_ROOT/lexicon/data/lexicon_entries/' 2>/dev/null || true"
do_cmd "cp -r '$ESDE_ROOT/integration/lexicon/expanded/'* '$LANG_ROOT/lexicon/data/expanded/' 2>/dev/null || true"
do_cmd "cp -r '$ESDE_ROOT/integration/lexicon/definitions/'* '$LANG_ROOT/lexicon/data/definitions/' 2>/dev/null || true"
do_cmd "cp -r '$ESDE_ROOT/integration/lexicon/mapper_output/'* '$LANG_ROOT/lexicon/data/mapper_output/' 2>/dev/null || true"

# Lexicon constitution
do_cmd "cp '$ESDE_ROOT/integration/lexicon/'*constitution* '$LANG_ROOT/lexicon/' 2>/dev/null || true"

# --- Projection ---
echo "  [projection] Experiment runner + eval"
do_cmd "cp '$ESDE_ROOT/scripts/run_projection_experiment.py' '$LANG_ROOT/projection/'"
do_cmd "cp '$ESDE_ROOT/tools/projection_eval.py' '$LANG_ROOT/projection/'"

# --- Sensor ---
echo "  [sensor] Phase 8 sensor modules"
do_cmd "cp '$ESDE_ROOT/sensor/'*.py '$LANG_ROOT/sensor/'"

# --- Relations ---
echo "  [relations] Relation pipeline"
do_cmd "cp '$ESDE_ROOT/integration/relations/'*.py '$LANG_ROOT/relations/'"

# --- Tests ---
echo "  [tests] Active tests"
do_cmd "cp '$ESDE_ROOT/tests/test_projection'*.py '$LANG_ROOT/tests/' 2>/dev/null || true"
do_cmd "cp '$ESDE_ROOT/tests/test_synapse'*.py '$LANG_ROOT/tests/' 2>/dev/null || true"
do_cmd "cp '$ESDE_ROOT/integration/lexicon/test_'*.py '$LANG_ROOT/tests/' 2>/dev/null || true"

# --- Docs ---
echo "  [docs] Active documentation"
do_cmd "cp '$ESDE_ROOT/Docs/ESDE_Glossary'*.md '$LANG_ROOT/docs/' 2>/dev/null || true"
do_cmd "cp '$ESDE_ROOT/Docs/ESDE_Module_Reference'*.md '$LANG_ROOT/docs/' 2>/dev/null || true"
do_cmd "cp '$ESDE_ROOT/Docs/ESDE_Essence'*.md '$LANG_ROOT/docs/' 2>/dev/null || true"
do_cmd "cp '$ESDE_ROOT/Docs/ESDE_Overview'*.md '$LANG_ROOT/docs/' 2>/dev/null || true"
do_cmd "cp '$ESDE_ROOT/Docs/ESDE_Experiment_Report'*.md '$LANG_ROOT/docs/experiment_reports/' 2>/dev/null || true"
do_cmd "cp '$ESDE_ROOT/Docs/ESDE_GPT_Audit'*.* '$LANG_ROOT/docs/experiment_reports/' 2>/dev/null || true"

# --- Cache (embedding files) ---
echo "  [cache] Embedding caches"
do_cmd "mkdir -p '$LANG_ROOT/cache'"
do_cmd "cp '$ESDE_ROOT/cache/'*.npz '$LANG_ROOT/cache/' 2>/dev/null || true"

echo ""

# ============================================================
# Step 3: Move completed modules → legacy/
# ============================================================
echo "--- Step 3: Move completed modules → legacy/ ---"

do_cmd "mkdir -p '$LEGACY_ROOT'"

FREEZE_DIRS=(
    esde_engine    # Phase 7 engine
    statistics     # Phase 9 W1-W5
    substrate      # Layer 0
    ledger         # Phase 8 hash chain
    index          # Phase 8 semantic index
    feedback       # Phase 8 feedback modulator
    pipeline       # Phase 8 core pipeline
    monitor        # Phase 8 TUI monitor
    runner         # Phase 8 long-run runner
    output_passive # Old passive outputs
)

for d in "${FREEZE_DIRS[@]}"; do
    if [ -d "$ESDE_ROOT/$d" ]; then
        echo "  Freeze: $d/"
        do_cmd "mv '$ESDE_ROOT/$d' '$LEGACY_ROOT/$d'"
    fi
done

# Lexicon freeze directories
LEXICON_FREEZE=(
    audit_output
    logs
    legacy
    report
    synapse_v4_report
)

do_cmd "mkdir -p '$LEGACY_ROOT/lexicon_archive'"
for d in "${LEXICON_FREEZE[@]}"; do
    if [ -d "$ESDE_ROOT/integration/lexicon/$d" ]; then
        echo "  Freeze: integration/lexicon/$d/"
        do_cmd "mv '$ESDE_ROOT/integration/lexicon/$d' '$LEGACY_ROOT/lexicon_archive/$d'"
    fi
done

echo ""

# ============================================================
# Step 4: Cleanup
# ============================================================
echo "--- Step 4: Cleanup ---"

echo "  Delete: __pycache__ directories"
do_cmd "find '$ESDE_ROOT' -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true"

echo ""

# ============================================================
# Summary
# ============================================================
echo "============================================================"
echo "Migration complete."
echo "============================================================"
echo ""
echo "New structure:"
echo "  $LANG_ROOT/"
if ! $DRY_RUN; then
    echo "    atoms/     : $(find $LANG_ROOT/atoms -type f 2>/dev/null | wc -l) files"
    echo "    synapse/   : $(find $LANG_ROOT/synapse -type f 2>/dev/null | wc -l) files"
    echo "    lexicon/   : $(find $LANG_ROOT/lexicon -type f 2>/dev/null | wc -l) files"
    echo "    projection/: $(find $LANG_ROOT/projection -type f 2>/dev/null | wc -l) files"
    echo "    sensor/    : $(find $LANG_ROOT/sensor -type f 2>/dev/null | wc -l) files"
    echo "    relations/ : $(find $LANG_ROOT/relations -type f 2>/dev/null | wc -l) files"
    echo "    tests/     : $(find $LANG_ROOT/tests -type f 2>/dev/null | wc -l) files"
    echo "    docs/      : $(find $LANG_ROOT/docs -type f 2>/dev/null | wc -l) files"
    echo "    cache/     : $(find $LANG_ROOT/cache -type f 2>/dev/null | wc -l) files"
fi
echo ""
echo "Frozen to legacy:"
echo "  $LEGACY_ROOT/"
echo ""
echo "Next steps:"
echo "  1. Verify language/ contents: ls -la $LANG_ROOT/*/"
echo "  2. Run tests from new location: cd $LANG_ROOT && python3 -m pytest tests/"
echo "  3. Update synapse_profiles.json paths"
echo "  4. Git commit both repos"
echo "  5. Optionally remove original esde/ copies after confirming"
