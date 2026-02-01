"""
ESDE Control Panel (Streamlit UI)
==================================

Minimal UI for Harvester + Pipeline operations.

Usage:
  cd /path/to/esde_phase9_v2
  streamlit run app.py

Design sources:
  - Gemini: "Control Panel" (URL Input, Harvest, Data Viewer, Policy Runner)
  - GPT: "最小UI（私が楽優先）" (Streamlit recommended)
"""

import streamlit as st
import subprocess
import sys
import os
import json
import time
from pathlib import Path

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
AXES = ["section", "passive", "paren", "quote", "propn", "section_passive"]


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
    st.markdown("*Phase 9 Weak Axis Statistics + Harvester*")
    st.markdown("---")
    
    # Status overview
    datasets = get_cached_datasets()
    has_output = (OUTPUT_DIR / "report.md").exists()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Cached Datasets", len(datasets))
    
    with col2:
        total_articles = sum(d["article_count"] for d in datasets)
        st.metric("Total Articles", total_articles)
    
    with col3:
        st.metric("Latest Output", "✅ Available" if has_output else "—")
    
    st.markdown("---")
    
    # Workflow guide
    st.markdown("### Workflow")
    
    st.markdown("""
    ```
    Step 1: 📥 Harvest  →  Fetch articles from Wikipedia (once)
    Step 2: ⚡ Pipeline  →  Run W2-W6 analysis (as many times as you want)
    Step 3: 📊 Results   →  View report, clustering, evidence
    ```
    """)
    
    if datasets:
        st.markdown("### Cached Datasets")
        for ds in datasets:
            st.markdown(f"- **{ds['name']}**: {ds['article_count']} articles ({ds['created_at'][:10]})")
    else:
        st.info("No datasets cached yet. Go to 📥 Harvest to get started.")


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
    st.markdown("*Run W2-W6 analysis on cached data*")
    st.markdown("---")
    
    datasets = get_cached_datasets()
    available = [d["name"] for d in datasets if d["article_count"] > 0]
    
    if not available:
        st.warning("No cached datasets with articles. Harvest first!")
        st.stop()
    
    # Configuration
    st.markdown("### Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dataset = st.selectbox("Dataset", available)
    
    with col2:
        axis = st.selectbox("Condition Axis", AXES, index=0)
    
    with col3:
        threshold = st.slider("Clustering Threshold", 0.5, 1.0, 0.9, 0.05)
    
    # Axis explanation
    axis_info = {
        "section": "Split by Wikipedia section names (Lead, Military campaigns, Death...)",
        "passive": "Split by passive voice vs active voice",
        "paren": "Split by inside/outside parentheses",
        "quote": "Split by inside/outside quotation marks",
        "propn": "Split by sentences containing proper nouns",
        "section_passive": "Combined: section × passive (more granular)",
    }
    st.caption(axis_info.get(axis, ""))
    
    st.markdown("---")
    
    # Run
    if st.button("🚀 Run Pipeline", type="primary", use_container_width=True):
        cmd = [
            sys.executable, "-m", "statistics.pipeline.run_full_pipeline",
            "--dataset", dataset,
            "--axis", axis,
            "--threshold", str(threshold),
        ]
        
        output_area = st.empty()
        with st.spinner(f"Running pipeline ({dataset}, axis={axis}, threshold={threshold})..."):
            rc, output = run_command(cmd, output_area)
        
        if rc == 0:
            st.success("Pipeline complete! Go to 📊 Results to view output.")
        else:
            st.error(f"Pipeline failed (exit code {rc})")


# ==========================================
# Page: Results
# ==========================================

elif page == "📊 Results":
    st.markdown("# 📊 Results")
    st.markdown("*View pipeline output*")
    st.markdown("---")
    
    report = load_output_report()
    analysis = load_output_json()
    stats = load_structure_stats()
    
    if not report and not analysis:
        st.info("No results yet. Run the pipeline first.")
        st.stop()
    
    tab1, tab2, tab3 = st.tabs(["📝 Report", "🏝️ Clustering", "📐 Structure Stats"])
    
    # Tab 1: Report
    with tab1:
        if report:
            st.markdown(report)
        else:
            st.info("No report.md found.")
    
    # Tab 2: Clustering
    with tab2:
        if analysis:
            w5 = analysis.get("w5_clustering", {})
            
            # Islands
            islands = w5.get("islands", [])
            noise = w5.get("noise", [])
            
            st.markdown(f"**Islands:** {len(islands)}, **Noise:** {len(noise)}")
            
            if islands:
                for i, island in enumerate(islands):
                    members = island.get("members", [])
                    cohesion = island.get("cohesion", 0)
                    st.markdown(f"### Island {i+1} (cohesion: {cohesion:.4f})")
                    
                    for m in members:
                        # Color by prefix
                        aid = m if isinstance(m, str) else m.get("article_id", str(m))
                        if aid.startswith("mil_"):
                            st.markdown(f"- 🗡️ `{aid}`")
                        elif aid.startswith("sch_"):
                            st.markdown(f"- 📚 `{aid}`")
                        elif aid.startswith("city_"):
                            st.markdown(f"- 🏙️ `{aid}`")
                        else:
                            st.markdown(f"- `{aid}`")
            
            if noise:
                st.markdown("### Noise (unclustered)")
                for aid in noise:
                    if isinstance(aid, str):
                        st.markdown(f"- `{aid}`")
                    else:
                        st.markdown(f"- `{aid}`")
            
            # Top resonances
            st.markdown("---")
            st.markdown("### Top Resonating Conditions")
            
            w3 = analysis.get("w3_axis_candidates", {})
            conditions = w3.get("conditions", {}) if isinstance(w3, dict) else {}
            
            if conditions:
                # Flatten and sort by max s-score
                all_scores = []
                for cond, data in conditions.items():
                    top_positive = data.get("top_positive", []) if isinstance(data, dict) else []
                    for item in top_positive[:3]:
                        if isinstance(item, dict):
                            all_scores.append((cond, item.get("token", "?"), item.get("s_score", 0)))
                        elif isinstance(item, (list, tuple)) and len(item) >= 2:
                            all_scores.append((cond, item[0], item[1]))
                
                all_scores.sort(key=lambda x: abs(x[2]), reverse=True)
                
                for cond, token, score in all_scores[:15]:
                    direction = "+" if score > 0 else "-"
                    st.markdown(f"- **{cond}**: `{token}` ({direction}{abs(score):.4f})")
        else:
            st.info("No analysis.json found.")
    
    # Tab 3: Structure Stats
    with tab3:
        if stats:
            st.json(stats)
        else:
            st.info("No structure_stats.json found.")


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
    
    # Tab 1: Distilled text
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
    
    # Tab 2: Substrate traces
    with tab2:
        meta = articles.get(selected_article, {})
        traces = meta.get("traces", {})
        
        if traces:
            st.json(traces)
        else:
            st.info("No traces available.")
    
    # Tab 3: Raw artifact
    with tab3:
        artifact_path = ARTIFACTS_DIR / f"{selected_article}.json"
        if artifact_path.exists():
            with open(artifact_path, "r") as f:
                artifact = json.load(f)
            
            # Show metadata (not full raw_json which is huge)
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
st.sidebar.markdown("*ESDE v5.4.8*")
st.sidebar.markdown("*Harvester v0.1.0*")
st.sidebar.markdown("*Aruism: Describe, but do not decide*")
