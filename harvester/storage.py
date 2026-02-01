"""
ESDE Harvester: Storage
========================

Two-layer storage following Gemini/GPT design:

  Layer A: Artifact (raw API response)
    → data/artifacts/<article_id>.json
    → "捨てるな" - preserve for future re-analysis
    
  Layer B: Distilled (extracted text)
    → data/datasets/<dataset_name>/<article_id>.txt
    → "記述せよ" - current analysis target
    
  Manifest: Dataset metadata
    → data/datasets/<dataset_name>/manifest.json
    → Substrate-compatible traces per article

Substrate alignment:
  - Traces use only approved namespaces (text:, meta:, wiki:, http:, fs:, harvest:)
  - No forbidden namespaces or key names
  - All values are machine-observable (INV-SUB-003)
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .fetcher import FetchResult


# ==========================================
# Constants
# ==========================================

DATA_ROOT = "data"
ARTIFACTS_DIR = os.path.join(DATA_ROOT, "artifacts")
DATASETS_DIR = os.path.join(DATA_ROOT, "datasets")

MANIFEST_FILENAME = "manifest.json"
MANIFEST_VERSION = "v0.1.0"

# Substrate-compatible trace keys for harvested articles
# Aligned with NAMESPACES.md v0.1.0 + new namespaces (http:, wiki:, fs:, harvest:)
HARVEST_CAPTURE_VERSION = "harvest_v0.1.0"


# ==========================================
# Artifact Storage (Layer A)
# ==========================================

def save_artifact(
    article_id: str,
    fetch_result: FetchResult,
    base_dir: str = ARTIFACTS_DIR,
) -> str:
    """
    Save raw API response as artifact.
    
    "捨てるな" - preserve raw data for future re-analysis.
    
    Args:
        article_id: Unique article identifier
        fetch_result: Raw fetch result
        base_dir: Artifacts directory
        
    Returns:
        Path to saved artifact file
    """
    os.makedirs(base_dir, exist_ok=True)
    
    artifact_path = os.path.join(base_dir, f"{article_id}.json")
    
    artifact = {
        "article_id": article_id,
        "title": fetch_result.title,
        "url": fetch_result.url,
        "status_code": fetch_result.status_code,
        "fetched_at": fetch_result.fetched_at,
        "raw_json": fetch_result.raw_json,
    }
    
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)
    
    return artifact_path


# ==========================================
# Distilled Storage (Layer B)
# ==========================================

def save_distilled(
    article_id: str,
    text: str,
    dataset_name: str,
    base_dir: str = DATASETS_DIR,
) -> str:
    """
    Save distilled text for analysis.
    
    "記述せよ" - current analysis target.
    
    Args:
        article_id: Unique article identifier
        text: Extracted plaintext
        dataset_name: Dataset name (e.g., "mixed")
        base_dir: Datasets directory
        
    Returns:
        Path to saved text file
    """
    dataset_dir = os.path.join(base_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    text_path = os.path.join(dataset_dir, f"{article_id}.txt")
    
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    return text_path


# ==========================================
# Substrate-Compatible Traces
# ==========================================

def compute_traces(
    article_id: str,
    fetch_result: FetchResult,
    artifact_path: str,
) -> Dict[str, Any]:
    """
    Compute Substrate-compatible traces from fetch result.
    
    All keys follow NAMESPACES.md conventions.
    All values are machine-observable (INV-SUB-003).
    No semantic interpretation (INV-SUB-002).
    
    Args:
        article_id: Article identifier
        fetch_result: Raw fetch result
        artifact_path: Path to saved artifact
        
    Returns:
        Dict of namespace:key → value traces
    """
    text = fetch_result.extract_text or ""
    
    traces = {
        # text: namespace (existing in NAMESPACES.md)
        "text:char_count": len(text),
        "text:word_count": len(text.split()),
        "text:line_count": text.count("\n") + 1 if text else 0,
        
        # meta: namespace (existing)
        "meta:domain": "en.wikipedia.org",
        
        # http: namespace (new - machine-observable)
        "http:status_code": fetch_result.status_code,
        
        # wiki: namespace (new - machine-observable, NO section_title per Cell Architecture §9.1)
        "wiki:section_count": fetch_result.section_count,
        
        # fs: namespace (new - file path reference)
        "fs:artifact_path": artifact_path,
        
        # harvest: namespace (new - collection metadata)
        "harvest:fetched_at": fetch_result.fetched_at,
        "harvest:api_version": "wikipedia_extracts_v1",
    }
    
    return traces


# ==========================================
# Manifest
# ==========================================

def load_manifest(dataset_name: str, base_dir: str = DATASETS_DIR) -> Optional[Dict]:
    """Load dataset manifest if it exists."""
    manifest_path = os.path.join(base_dir, dataset_name, MANIFEST_FILENAME)
    
    if not os.path.exists(manifest_path):
        return None
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(
    dataset_name: str,
    articles: Dict[str, Dict[str, Any]],
    base_dir: str = DATASETS_DIR,
) -> str:
    """
    Save dataset manifest.
    
    Args:
        dataset_name: Dataset name
        articles: Dict of article_id → metadata
        base_dir: Datasets directory
        
    Returns:
        Path to manifest file
    """
    dataset_dir = os.path.join(base_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_name": dataset_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "article_count": len(articles),
        "articles": articles,
    }
    
    manifest_path = os.path.join(dataset_dir, MANIFEST_FILENAME)
    
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    return manifest_path


# ==========================================
# Dataset Reader (for pipeline)
# ==========================================

def load_dataset(dataset_name: str, base_dir: str = DATASETS_DIR) -> Dict[str, str]:
    """
    Load distilled texts from a cached dataset.
    
    Args:
        dataset_name: Dataset name
        base_dir: Datasets directory
        
    Returns:
        Dict of article_id → text content
        
    Raises:
        FileNotFoundError: If dataset not found
    """
    manifest = load_manifest(dataset_name, base_dir)
    
    if manifest is None:
        dataset_dir = os.path.join(base_dir, dataset_name)
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found at {dataset_dir}/\n"
            f"Run: python -m harvester.cli harvest --dataset {dataset_name}"
        )
    
    articles = {}
    dataset_dir = os.path.join(base_dir, dataset_name)
    
    for article_id, meta in manifest["articles"].items():
        text_path = os.path.join(dataset_dir, f"{article_id}.txt")
        
        if not os.path.exists(text_path):
            print(f"  [!] Missing text file: {text_path}")
            continue
        
        with open(text_path, "r", encoding="utf-8") as f:
            articles[article_id] = f.read()
    
    return articles


def list_datasets(base_dir: str = DATASETS_DIR) -> List[Dict[str, Any]]:
    """
    List all cached datasets.
    
    Returns:
        List of dataset info dicts
    """
    if not os.path.exists(base_dir):
        return []
    
    results = []
    
    for name in sorted(os.listdir(base_dir)):
        dataset_dir = os.path.join(base_dir, name)
        if not os.path.isdir(dataset_dir):
            continue
        
        manifest = load_manifest(name, base_dir)
        if manifest:
            results.append({
                "name": name,
                "article_count": manifest.get("article_count", 0),
                "created_at": manifest.get("created_at", "unknown"),
            })
        else:
            # Count .txt files
            txt_count = len([f for f in os.listdir(dataset_dir) if f.endswith(".txt")])
            results.append({
                "name": name,
                "article_count": txt_count,
                "created_at": "no manifest",
            })
    
    return results
