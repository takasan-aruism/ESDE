"""
ESDE Control Panel (Streamlit UI)
==================================

Phase 9-UI v0: Observation Renderer

UI is a "viewer" — reads output/ only, never writes or computes.
Phase 9 = observation engine (JSON/MD/CSV generator)
Phase 9-UI = observation display (renders existing data)

Usage:
  cd /path/to/esde_phase9_v2
  streamlit run app.py

Design sources:
  - Gemini: "Control Panel" (URL Input, Harvest, Data Viewer, Policy Runner)
  - GPT: "Phase 9-UI v0" (k-sweep, threshold trace, islands explorer)
  - Claude: Implementation
"""

import streamlit as st
import subprocess
import sys
import os
import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# ==========================================
# Config
# ==========================================

st.set_page_config(
    page_title="ESDE Control Panel",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
DATA_ROOT = Path("data")
ARTIFACTS_DIR = DATA_ROOT / "artifacts"
DATASETS_DIR = DATA_ROOT / "datasets"
OUTPUT_DIR = Path("output")

# Available axes
AXES = ["section", "document", "passive", "paren", "quote", "propn", "section_passive"]


# ==========================================
# Styling
# ==========================================

st.markdown("""
<style>
    .stApp { max-width: 1200px; margin: 0 auto; }
    .metric-card {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 16px;
        border-left: 3px solid #89b4fa;
    }
    .success-box {
        background: #1e3a2f;
        border-left: 3px solid #a6e3a1;
        padding: 12px;
        border-radius: 4px;
        margin: 8px 0;
    }
    .warn-box {
        background: #3a2e1e;
        border-left: 3px solid #f9e2af;
        padding: 12px;
        border-radius: 4px;
        margin: 8px 0;
    }
    div[data-testid="stSidebar"] { min-width: 300px; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# Helper Functions
# ==========================================

def get_cached_datasets():
    """List cached datasets from data/datasets/."""
    if not DATASETS_DIR.exists():
        return []
    
    results = []
    for d in sorted(DATASETS_DIR.iterdir()):
        if not d.is_dir():
            continue
        manifest_path = d / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            results.append({
                "name": d.name,
                "article_count": manifest.get("article_count", 0),
                "created_at": manifest.get("created_at", "unknown"),
                "articles": manifest.get("articles", {}),
            })
        else:
            txt_count = len(list(d.glob("*.txt")))
            if txt_count > 0:
                results.append({
                    "name": d.name,
                    "article_count": txt_count,
                    "created_at": "no manifest",
                    "articles": {},
                })
    return results


def load_manifest(dataset_name):
    """Load manifest for a dataset."""
    path = DATASETS_DIR / dataset_name / "manifest.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def load_output_report():
    """Load the latest report.md."""
    path = OUTPUT_DIR / "report.md"
    if path.exists():
        with open(path, "r") as f:
            return f.read()
    return None


def load_output_json():
    """Load the latest analysis.json."""
    path = OUTPUT_DIR / "analysis.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def load_structure_stats():
    """Load structure_stats.json."""
    path = OUTPUT_DIR / "structure_stats.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def load_k_sweep_csv():
    """Load k_sweep.csv if exists."""
    path = OUTPUT_DIR / "k_sweep.csv"
    if not path.exists():
        return None
    
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def run_command(cmd, placeholder):
    """Run a command and stream output to placeholder."""
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    output_lines = []
    for line in process.stdout:
        output_lines.append(line)
        placeholder.code("".join(output_lines), language="text")
    
    process.wait()
    return process.returncode, "".join(output_lines)


def member_icon(member_id: str) -> str:
    """Return an icon based on article prefix."""
    if "__" in member_id:
        article_part = member_id.split("__")[0]
    else:
        article_part = member_id
    
    if article_part.startswith("mil_"):
        return "🗡️"
    elif article_part.startswith("sch_"):
        return "📚"
    elif article_part.startswith("city_"):
        return "🏙️"
    else:
        return "📄"


def format_member_label(member_id: str) -> str:
    """Format member ID for display: article__section -> readable form."""
    if "__" in member_id:
        article, section = member_id.split("__", 1)
        return f"{article} > {section.replace('_', ' ')}"
    return member_id


# ==========================================
# Sidebar: Navigation
# ==========================================

st.sidebar.markdown("# 🔬 ESDE")
st.sidebar.markdown("**Control Panel**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Dashboard", "📥 Harvest", "⚡ Pipeline", "📊 Results", "📦 Data Viewer"],
    label_visibility="collapsed",
)


# ==========================================
# Page: Dashboard
# ==========================================

if page == "🏠 Dashboard":
    st.markdown("# 🔬 ESDE Control Panel")
    st.markdown("*Phase 9 Weak Axis Statistics — Observation Renderer*")
    st.markdown("---")
    
    # Status overview
    datasets = get_cached_datasets()
    analysis = load_output_json()
    has_output = analysis is not None
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Cached Datasets", len(datasets))
    
    with col2:
        total_articles = sum(d["article_count"] for d in datasets)
        st.metric("Total Articles", total_articles)
    
    with col3:
        if analysis:
            lens = analysis.get("lens", "—")
            st.metric("Last Lens", lens)
        else:
            st.metric("Last Lens", "—")
    
    with col4:
        st.metric("Latest Output", "✅" if has_output else "—")
    
    st.markdown("---")
    
    # Workflow guide
    st.markdown("### Workflow")
    
    st.markdown("""
    ```
    Step 1: 📥 Harvest   →  Fetch articles from Wikipedia (once)
    Step 2: ⚡ Pipeline   →  Run analysis with Lens + Threshold + Edge Policy
    Step 3: 📊 Results    →  Explore k-sweep, threshold trace, islands
    ```
    """)
    
    if datasets:
        st.markdown("### Cached Datasets")
        for ds in datasets:
            st.markdown(f"- **{ds['name']}**: {ds['article_count']} articles ({ds['created_at'][:10]})")
    else:
        st.info("No datasets cached yet. Go to 📥 Harvest to get started.")
    
    # Quick summary of last run
    if analysis:
        st.markdown("---")
        st.markdown("### Last Run Summary")
        
        w5 = analysis.get("w5_clustering", {})
        chaining = analysis.get("chaining_metrics", {})
        tt = analysis.get("threshold_trace", {})
        
        cols = st.columns(4)
        with cols[0]:
            st.metric("Islands", w5.get("island_count", "—"))
        with cols[1]:
            st.metric("Noise", w5.get("noise_count", "—"))
        with cols[2]:
            gcr = chaining.get("giant_component_ratio")
            st.metric("GCR", f"{gcr:.3f}" if gcr is not None else "—")
        with cols[3]:
            t_res = tt.get("t_resolved") or analysis.get("threshold")
            st.metric("Threshold", f"{t_res:.4f}" if t_res else "—")


# ==========================================
# Page: Harvest
# ==========================================

elif page == "📥 Harvest":
    st.markdown("# 📥 Harvest")
    st.markdown("*Fetch articles from Wikipedia and cache locally*")
    st.markdown("---")
    
    # Built-in datasets
    st.markdown("### Built-in Datasets")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**mixed** (15 articles)")
        st.markdown("Military leaders (5) + Scholars (5) + Cities (5)")
        force_mixed = st.checkbox("Force re-harvest", key="force_mixed")
        if st.button("🌐 Harvest Mixed", use_container_width=True):
            cmd = [sys.executable, "-m", "harvester.cli", "harvest", "--dataset", "mixed"]
            if force_mixed:
                cmd.append("--force")
            
            output_area = st.empty()
            with st.spinner("Fetching from Wikipedia..."):
                rc, output = run_command(cmd, output_area)
            
            if rc == 0:
                st.success("Harvest complete!")
            else:
                st.error(f"Harvest failed (exit code {rc})")
    
    with col2:
        st.markdown("**warlords** (10 articles)")
        st.markdown("Sengoku era warlords (Nobunaga, Hideyoshi, Ieyasu...)")
        force_warlords = st.checkbox("Force re-harvest", key="force_warlords")
        if st.button("🌐 Harvest Warlords", use_container_width=True):
            cmd = [sys.executable, "-m", "harvester.cli", "harvest", "--dataset", "warlords"]
            if force_warlords:
                cmd.append("--force")
            
            output_area = st.empty()
            with st.spinner("Fetching from Wikipedia..."):
                rc, output = run_command(cmd, output_area)
            
            if rc == 0:
                st.success("Harvest complete!")
            else:
                st.error(f"Harvest failed (exit code {rc})")
    
    st.markdown("---")
    
    # Cached datasets
    st.markdown("### Cached Datasets")
    datasets = get_cached_datasets()
    
    if datasets:
        for ds in datasets:
            with st.expander(f"📁 {ds['name']} ({ds['article_count']} articles)"):
                st.markdown(f"**Created:** {ds['created_at']}")
                
                if ds["articles"]:
                    # Group by prefix
                    groups = {}
                    for aid, meta in ds["articles"].items():
                        prefix = meta.get("prefix", "unknown")
                        if prefix not in groups:
                            groups[prefix] = []
                        groups[prefix].append((aid, meta))
                    
                    for prefix, items in sorted(groups.items()):
                        st.markdown(f"**[{prefix}]** ({len(items)} articles)")
                        for aid, meta in items:
                            traces = meta.get("traces", {})
                            chars = traces.get("text:char_count", 0)
                            sections = traces.get("wiki:section_count", 0)
                            st.markdown(f"- `{aid}`: {chars:,} chars, {sections} sections")
    else:
        st.info("No datasets cached. Use the buttons above to harvest.")


# ==========================================
# Page: Pipeline
# ==========================================

elif page == "⚡ Pipeline":
    st.markdown("# ⚡ Pipeline")
    st.markdown("*Run Phase 9 analysis on cached data*")
    st.markdown("---")
    
    datasets = get_cached_datasets()
    available = [d["name"] for d in datasets if d["article_count"] > 0]
    
    if not available:
        st.warning("No cached datasets with articles. Harvest first!")
        st.stop()
    
    # -- Row 1: Dataset + Lens --
    st.markdown("### Data & Lens")
    col1, col2 = st.columns(2)
    
    with col1:
        dataset = st.selectbox("Dataset", available)
    
    with col2:
        lens_mode = st.selectbox(
            "Lens",
            ["🔀 Hybrid", "🔬 Structure", "🧬 Semantic", "⚙️ Custom (axis only)"],
        )
    
    lens_map = {
        "🔬 Structure": "structure",
        "🧬 Semantic": "semantic",
        "🔀 Hybrid": "hybrid",
        "⚙️ Custom (axis only)": None,
    }
    selected_lens = lens_map[lens_mode]
    
    # Show axis selector only in custom mode
    if selected_lens is None:
        axis = st.selectbox("Condition Axis", AXES, index=0)
        axis_info = {
            "section": "Split by Wikipedia section names",
            "document": "Split by document (article) name",
            "passive": "Split by passive voice vs active voice",
            "paren": "Split by inside/outside parentheses",
            "quote": "Split by inside/outside quotation marks",
            "propn": "Split by sentences containing proper nouns",
            "section_passive": "Combined: section x passive",
        }
        st.caption(axis_info.get(axis, ""))
    else:
        lens_desc = {
            "structure": "📐 Section x token frequency → Wikipedia template topology",
            "semantic": "🧬 Document x 20-dim vector → subject matter clustering",
            "hybrid": "🔀 Section x 20-dim vector → semantic bias within sections",
        }
        st.info(lens_desc.get(selected_lens, ""))
        axis = None
    
    st.markdown("---")
    
    # -- Row 2: Threshold --
    st.markdown("### Threshold")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        threshold_mode = st.selectbox("Mode", ["quantile (dynamic)", "fixed (legacy)"])
    
    with col2:
        if "quantile" in threshold_mode:
            threshold_q = st.slider("Quantile (q)", 0.50, 1.00, 0.98, 0.01,
                                     help="0.98 = top 2% similarity pairs define threshold")
            threshold_val = 0.9
        else:
            threshold_val = st.slider("Fixed Threshold", 0.50, 1.00, 0.90, 0.05)
            threshold_q = 0.98
    
    with col3:
        min_island = st.number_input("Min Island Size", 2, 20, 2)
    
    st.markdown("---")
    
    # -- Row 3: Edge Policy --
    st.markdown("### Edge Policy")
    col1, col2 = st.columns(2)
    
    with col1:
        edge_filter = st.selectbox(
            "Edge Filter",
            ["mutual_knn (anti-chaining)", "none (single-linkage)"],
        )
    
    with col2:
        if "mutual_knn" in edge_filter:
            knn_mode = st.selectbox(
                "k Selection",
                ["auto (k-sweep)", "fixed"],
            )
            if "auto" in knn_mode:
                knn_k_val = "auto"
            else:
                knn_k_val = str(st.number_input("k value", 2, 20, 3))
        else:
            knn_k_val = None
    
    # Advanced options
    force_extract = st.checkbox("Force W1 re-extraction (ignore cache)", value=False,
                                 help="Normally W1 features are cached after first run. Check this to force spaCy re-processing.")
    
    st.markdown("---")
    
    # -- Build command --
    cmd = [
        sys.executable, "-m", "statistics.pipeline.run_full_pipeline",
        "--dataset", dataset,
        "--min-island", str(min_island),
    ]
    
    if selected_lens:
        cmd.extend(["--lens", selected_lens])
    else:
        cmd.extend(["--axis", axis])
    
    if "quantile" in threshold_mode:
        cmd.extend(["--threshold-mode", "quantile", "--threshold-q", str(threshold_q)])
    else:
        cmd.extend(["--threshold", str(threshold_val)])
    
    if "mutual_knn" in edge_filter:
        cmd.extend(["--edge-filter", "mutual_knn"])
        if knn_k_val:
            cmd.extend(["--knn-k", knn_k_val])
    else:
        cmd.extend(["--edge-filter", "none"])
    
    if force_extract:
        cmd.append("--force-extract")
    
    # Show command preview
    cmd_str = " ".join(cmd[1:])
    st.code(cmd_str, language="bash")
    
    # Run
    if st.button("🚀 Run Pipeline", type="primary", use_container_width=True):
        label_parts = []
        if selected_lens:
            label_parts.append(f"lens={selected_lens}")
        else:
            label_parts.append(f"axis={axis}")
        if "mutual_knn" in edge_filter:
            label_parts.append(f"knn-k={knn_k_val}")
        label = ", ".join(label_parts)
        
        output_area = st.empty()
        with st.spinner(f"Running pipeline ({dataset}, {label})..."):
            rc, output = run_command(cmd, output_area)
        
        if rc == 0:
            st.success("Pipeline complete! Go to 📊 Results to view output.")
        else:
            st.error(f"Pipeline failed (exit code {rc})")


# ==========================================
# Page: Results (Phase 9-UI v0)
# ==========================================

elif page == "📊 Results":
    st.markdown("# 📊 Results")
    st.markdown("*Phase 9 Observation Renderer*")
    st.markdown("---")
    
    analysis = load_output_json()
    k_sweep_data = load_k_sweep_csv()
    stats = load_structure_stats()
    report = load_output_report()
    
    if not analysis:
        st.info("No results yet. Run the pipeline first.")
        st.stop()
    
    # -- Header: Run summary --
    lens_name = analysis.get("lens", analysis.get("feature_mode", "unknown"))
    axis_name = analysis.get("axis", "—")
    w5 = analysis.get("w5_clustering", {})
    chaining = analysis.get("chaining_metrics", {})
    tt = analysis.get("threshold_trace", {})
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Lens", lens_name)
    with col2:
        st.metric("Islands", w5.get("island_count", "—"))
    with col3:
        st.metric("Noise", w5.get("noise_count", "—"))
    with col4:
        gcr = chaining.get("giant_component_ratio")
        gcr_label = f"{gcr:.3f}" if gcr is not None else "—"
        st.metric("GCR", gcr_label)
    with col5:
        t_res = tt.get("t_resolved") or analysis.get("threshold")
        st.metric("Threshold", f"{t_res:.4f}" if t_res else "—")
    
    st.markdown("---")
    
    # -- Tabs --
    tab1, tab2, tab3, tab4 = st.tabs(["📈 k-sweep", "🎚️ Threshold", "🏝️ Islands", "📝 Report"])
    
    # ────────────────────────────────────────
    # Tab 1: k-sweep
    # ────────────────────────────────────────
    with tab1:
        if not k_sweep_data:
            st.info("No k-sweep data. Run pipeline with `--edge-filter mutual_knn --knn-k auto` to generate.")
            
            # Still show edge policy trace if available
            ept = analysis.get("edge_policy_trace", {})
            if ept:
                st.markdown("### Edge Policy Trace")
                st.json(ept)
        else:
            st.markdown("### k-sweep Results")
            st.markdown("*Each row shows clustering at a different focal length (k). "
                        "The transition from small islands to a giant component is the percolation threshold.*")
            
            # Parse CSV data (column names from edge_policy.export_sweep_csv)
            sweep_rows = []
            for row in k_sweep_data:
                mi = row.get("mean_intra_sim", "")
                sweep_rows.append({
                    "k": int(row.get("k", 0)),
                    "edges": int(row.get("n_edges", 0)),
                    "islands": int(row.get("n_islands", 0)),
                    "noise": int(row.get("n_noise", 0)),
                    "largest": int(row.get("largest_island", 0)),
                    "gcr": float(row.get("giant_ratio", 0)),
                    "mean_intra": float(mi) if mi else 0.0,
                    "ok": str(row.get("satisfies_policy", "")).strip().lower() == "true",
                })
            
            # Chosen k marker
            ept = analysis.get("edge_policy_trace", {})
            k_chosen = ept.get("k_chosen")
            
            # Table
            table_md = "| k | edges | islands | noise | largest | gcr | mean_intra | ok |\n"
            table_md += "|--:|------:|--------:|------:|--------:|----:|-----------:|:--:|\n"
            for r in sweep_rows:
                marker = " **←**" if r["k"] == k_chosen else ""
                ok_mark = "✓" if r["ok"] else ""
                table_md += (f"| {r['k']}{marker} | {r['edges']} | {r['islands']} | {r['noise']} | "
                             f"{r['largest']} | {r['gcr']:.4f} | {r['mean_intra']:.4f} | {ok_mark} |\n")
            
            st.markdown(table_md)
            
            # -- Charts --
            st.markdown("---")
            
            try:
                import altair as alt
                import pandas as pd
                
                df = pd.DataFrame(sweep_rows)
                
                # Chart 1: GCR
                st.markdown("#### Giant Component Ratio (gcr)")
                st.markdown("*The jump marks the percolation threshold — beyond this k, chaining dominates.*")
                
                gcr_chart = alt.Chart(df).mark_bar(
                    color="#f38ba8",
                    cornerRadiusTopLeft=3,
                    cornerRadiusTopRight=3,
                ).encode(
                    x=alt.X("k:O", title="k (focal length)"),
                    y=alt.Y("gcr:Q", title="Giant Component Ratio", scale=alt.Scale(domain=[0, 1])),
                    opacity=alt.condition(
                        alt.datum.ok == True,
                        alt.value(1.0),
                        alt.value(0.4),
                    ),
                    tooltip=["k", "gcr", "largest", "islands", "noise"],
                )
                
                policy_line = alt.Chart(pd.DataFrame({"y": [0.20]})).mark_rule(
                    strokeDash=[5, 3], color="#a6adc8",
                ).encode(y="y:Q")
                
                st.altair_chart(gcr_chart + policy_line, use_container_width=True)
                
                # Chart 2: Mean Intra-Similarity
                st.markdown("#### Mean Intra-Similarity")
                st.markdown("*Higher = tighter clusters. Drops as k increases and islands merge.*")
                
                intra_chart = alt.Chart(df).mark_bar(
                    color="#89b4fa",
                    cornerRadiusTopLeft=3,
                    cornerRadiusTopRight=3,
                ).encode(
                    x=alt.X("k:O", title="k (focal length)"),
                    y=alt.Y("mean_intra:Q", title="Mean Intra-Similarity"),
                    opacity=alt.condition(
                        alt.datum.ok == True,
                        alt.value(1.0),
                        alt.value(0.4),
                    ),
                    tooltip=["k", "mean_intra", "islands", "largest"],
                )
                
                policy_line2 = alt.Chart(pd.DataFrame({"y": [0.25]})).mark_rule(
                    strokeDash=[5, 3], color="#a6adc8",
                ).encode(y="y:Q")
                
                st.altair_chart(intra_chart + policy_line2, use_container_width=True)
                
                # Chart 3: Island count + Noise
                st.markdown("#### Islands & Noise by k")
                
                melt_df = df[["k", "islands", "noise"]].melt(
                    id_vars=["k"], var_name="category", value_name="count"
                )
                
                stacked = alt.Chart(melt_df).mark_bar(
                    cornerRadiusTopLeft=2,
                    cornerRadiusTopRight=2,
                ).encode(
                    x=alt.X("k:O", title="k"),
                    y=alt.Y("count:Q", title="Count"),
                    color=alt.Color("category:N",
                        scale=alt.Scale(
                            domain=["islands", "noise"],
                            range=["#a6e3a1", "#6c7086"],
                        ),
                        legend=alt.Legend(title=""),
                    ),
                    tooltip=["k", "category", "count"],
                )
                st.altair_chart(stacked, use_container_width=True)
                
            except ImportError:
                st.warning("Install `altair` and `pandas` for charts: `pip install altair pandas`")
    
    # ────────────────────────────────────────
    # Tab 2: Threshold Trace
    # ────────────────────────────────────────
    with tab2:
        if not tt or tt.get("mode") == "fixed":
            # Fixed mode or no trace
            threshold_val = tt.get("t_resolved") if tt else analysis.get("threshold")
            if threshold_val:
                st.metric("Threshold Used", f"{threshold_val:.4f}")
                if tt:
                    st.caption(f"Mode: {tt.get('mode', 'fixed')}")
            
            if not tt:
                st.info("No threshold trace. Run pipeline with `--threshold-mode quantile` for full trace.")
            
            # Still show chaining if available
            if chaining:
                st.markdown("---")
                st.markdown("#### Chaining Diagnostics")
                _show_chaining_metrics(chaining) if False else None  # defined inline below
        
        if tt and tt.get("mode") != "fixed":
            st.markdown("### Threshold Trace")
            st.markdown("*How the 3-layer dynamic threshold was resolved. "
                        '"Describe, but do not decide" — the resolver shows its reasoning.*')
            
            # 3 values side by side
            col1, col2, col3 = st.columns(3)
            
            t_abs = tt.get("t_abs")
            t_rel = tt.get("t_rel")
            t_resolved = tt.get("t_resolved")
            
            with col1:
                st.markdown("##### t_abs (absolute floor)")
                if t_abs is not None:
                    st.markdown(f"### `{t_abs:.4f}`")
                    
                    abs_source = tt.get("abs_source", {})
                    source_type = abs_source.get("source", "unknown")
                    if source_type == "global_model":
                        n = abs_source.get("n_pairs", 0)
                        q = abs_source.get("quantile_q", "?")
                        st.caption(f"Global model (n={n:,}, q={q})")
                    elif source_type == "lens_floor":
                        floor = abs_source.get("floor", "?")
                        st.caption(f"Lens floor = {floor}")
                    elif source_type == "fallback":
                        st.caption("Fallback (insufficient global data)")
                    else:
                        st.caption(f"Source: {source_type}")
                else:
                    st.markdown("### `—`")
            
            with col2:
                st.markdown("##### t_rel (relative / this run)")
                if t_rel is not None:
                    st.markdown(f"### `{t_rel:.4f}`")
                    q = tt.get("quantile_q", "?")
                    st.caption(f"Q({q}) of this run's similarities")
                else:
                    st.markdown("### `—`")
            
            with col3:
                st.markdown("##### t_resolved ✅")
                if t_resolved is not None:
                    st.markdown(f"### `{t_resolved:.4f}`")
                    strategy = tt.get("resolve_strategy", "safety_first")
                    st.caption(f"max(t_abs, t_rel, floor) — {strategy}")
                else:
                    st.markdown("### `—`")
            
            st.markdown("---")
            
            # Similarity distribution
            dist = tt.get("run_distribution", {})
            if dist:
                st.markdown("#### Similarity Distribution (this run)")
                
                dist_cols = st.columns(5)
                for i, (key, label) in enumerate([
                    ("min", "Min"), ("q25", "Q25"), ("median", "Median"),
                    ("q75", "Q75"), ("max", "Max"),
                ]):
                    val = dist.get(key)
                    with dist_cols[i]:
                        st.metric(label, f"{val:.4f}" if val is not None else "—")
                
                n_pairs = dist.get("count", dist.get("n_pairs"))
                mean = dist.get("mean")
                std = dist.get("std")
                if n_pairs or mean:
                    extra_cols = st.columns(3)
                    with extra_cols[0]:
                        st.metric("Pairs", f"{n_pairs:,}" if n_pairs else "—")
                    with extra_cols[1]:
                        st.metric("Mean", f"{mean:.4f}" if mean is not None else "—")
                    with extra_cols[2]:
                        st.metric("Std", f"{std:.4f}" if std is not None else "—")
        
        # Chaining metrics (always show if available)
        if chaining:
            st.markdown("---")
            st.markdown("#### Chaining Diagnostics")
            
            ch_cols = st.columns(4)
            with ch_cols[0]:
                gcr_val = chaining.get("giant_component_ratio", 0)
                delta_label = "healthy" if gcr_val <= 0.20 else "chaining risk"
                delta_color = "normal" if gcr_val <= 0.20 else "off"
                st.metric("GCR", f"{gcr_val:.4f}", delta=delta_label, delta_color=delta_color)
            with ch_cols[1]:
                st.metric("Largest Island", chaining.get("largest_island_size", "—"))
            with ch_cols[2]:
                sparsity = chaining.get("edge_sparsity")
                st.metric("Edge Sparsity", f"{sparsity:.4f}" if sparsity is not None else "—")
            with ch_cols[3]:
                mean_intra = chaining.get("mean_intra_similarity")
                st.metric("Mean Intra-Sim", f"{mean_intra:.4f}" if mean_intra is not None else "—")
            
            if chaining.get("chaining_detected"):
                st.warning("⚠️ Chaining detected — consider using Mutual-kNN edge filter.")
        
        # Raw trace
        if tt:
            with st.expander("Raw threshold trace (JSON)"):
                st.json(tt)
    
    # ────────────────────────────────────────
    # Tab 3: Islands Explorer
    # ────────────────────────────────────────
    with tab3:
        islands = w5.get("islands", [])
        noise = w5.get("noise", w5.get("noise_ids", []))
        input_count = w5.get("input_count", 0)
        
        if not islands and not noise:
            st.info("No clustering results found.")
        else:
            st.markdown("### Islands Explorer")
            st.markdown(f"**{len(islands)} islands**, **{len(noise)} noise** out of **{input_count} sections**")
            
            # Search
            search_term = st.text_input("🔍 Search members", placeholder="Type to filter (e.g. 'early_life', 'tokyo', 'mil_')...")
            
            st.markdown("---")
            
            # Island list
            for i, island in enumerate(sorted(islands, key=lambda x: x.get("size", 0), reverse=True)):
                members = island.get("members", island.get("member_ids", []))
                size = island.get("size", len(members))
                cohesion = island.get("cohesion", island.get("cohesion_score", 0))
                
                # Filter by search
                if search_term:
                    matching = [m for m in members if search_term.lower() in m.lower()]
                    if not matching:
                        continue
                else:
                    matching = members
                
                # Island header
                header = f"🏝️ Island {i+1} — {size} members, cohesion {cohesion:.4f}"
                if search_term:
                    header += f" ({len(matching)}/{size} match)"
                
                with st.expander(header, expanded=(i == 0 and not search_term)):
                    # Group by article
                    by_article = {}
                    for m in matching:
                        if "__" in m:
                            article = m.split("__")[0]
                        else:
                            article = m
                        if article not in by_article:
                            by_article[article] = []
                        by_article[article].append(m)
                    
                    for article, article_members in sorted(by_article.items()):
                        icon = member_icon(article)
                        st.markdown(f"**{icon} {article}** ({len(article_members)})")
                        for m in sorted(article_members):
                            label = format_member_label(m)
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;`{label}`")
            
            # Noise section
            if noise:
                st.markdown("---")
                
                if search_term:
                    matching_noise = [n_id for n_id in noise if search_term.lower() in str(n_id).lower()]
                else:
                    matching_noise = noise
                
                if matching_noise:
                    header = f"🌫️ Noise — {len(noise)} unclustered"
                    if search_term:
                        header += f" ({len(matching_noise)} match)"
                    
                    with st.expander(header, expanded=False):
                        by_article = {}
                        for n_id in matching_noise:
                            if isinstance(n_id, str):
                                if "__" in n_id:
                                    article = n_id.split("__")[0]
                                else:
                                    article = n_id
                                if article not in by_article:
                                    by_article[article] = []
                                by_article[article].append(n_id)
                        
                        for article, article_members in sorted(by_article.items()):
                            icon = member_icon(article)
                            st.markdown(f"**{icon} {article}** ({len(article_members)})")
                            for m in sorted(article_members):
                                label = format_member_label(m)
                                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;`{label}`")
    
    # ────────────────────────────────────────
    # Tab 4: Report (raw)
    # ────────────────────────────────────────
    with tab4:
        if report:
            st.markdown(report)
        else:
            st.info("No report.md found.")
        
        st.markdown("---")
        
        with st.expander("Raw analysis.json"):
            st.json(analysis)
        
        if stats:
            with st.expander("Structure statistics"):
                st.json(stats)


# ==========================================
# Page: Data Viewer
# ==========================================

elif page == "📦 Data Viewer":
    st.markdown("# 📦 Data Viewer")
    st.markdown("*Browse cached articles and artifacts*")
    st.markdown("---")
    
    datasets = get_cached_datasets()
    
    if not datasets:
        st.info("No datasets cached.")
        st.stop()
    
    dataset_names = [d["name"] for d in datasets]
    selected_ds = st.selectbox("Dataset", dataset_names)
    
    manifest = load_manifest(selected_ds)
    if not manifest:
        st.warning("No manifest found for this dataset.")
        st.stop()
    
    articles = manifest.get("articles", {})
    article_ids = sorted(articles.keys())
    
    if not article_ids:
        st.info("No articles in this dataset.")
        st.stop()
    
    selected_article = st.selectbox("Article", article_ids)
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📄 Text", "🔍 Traces", "📦 Raw Artifact"])
    
    with tab1:
        text_path = DATASETS_DIR / selected_ds / f"{selected_article}.txt"
        if text_path.exists():
            with open(text_path, "r") as f:
                text = f.read()
            
            st.markdown(f"**Characters:** {len(text):,}")
            st.markdown(f"**Words:** {len(text.split()):,}")
            st.markdown(f"**Lines:** {text.count(chr(10)) + 1}")
            st.markdown("---")
            st.text_area("Content", text, height=400, label_visibility="collapsed")
        else:
            st.warning(f"Text file not found: {text_path}")
    
    with tab2:
        meta = articles.get(selected_article, {})
        traces = meta.get("traces", {})
        
        if traces:
            st.json(traces)
        else:
            st.info("No traces available.")
    
    with tab3:
        artifact_path = ARTIFACTS_DIR / f"{selected_article}.json"
        if artifact_path.exists():
            with open(artifact_path, "r") as f:
                artifact = json.load(f)
            
            display = {k: v for k, v in artifact.items() if k != "raw_json"}
            st.json(display)
            
            if st.checkbox("Show full raw API response"):
                st.json(artifact.get("raw_json", {}))
        else:
            st.info(f"Artifact not found: {artifact_path}")


# ==========================================
# Footer
# ==========================================

st.sidebar.markdown("---")
st.sidebar.markdown("*ESDE v5.5.0*")
st.sidebar.markdown("*Harvester v0.1.0*")
st.sidebar.markdown("*Phase 9-UI v0*")
st.sidebar.markdown('*"Describe, but do not decide"*')
