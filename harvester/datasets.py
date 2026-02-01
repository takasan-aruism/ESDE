"""
ESDE Harvester: Dataset Definitions
====================================

Centralized dataset definitions.
Moved from run_full_pipeline.py for reuse.
"""

from typing import Dict, List


# ==========================================
# Dataset: Sengoku Warlords (10)
# ==========================================

SENGOKU_WARLORDS: Dict[str, List[str]] = {
    "mil": [
        "Oda Nobunaga",
        "Toyotomi Hideyoshi",
        "Tokugawa Ieyasu",
        "Takeda Shingen",
        "Uesugi Kenshin",
        "Date Masamune",
        "Sanada Yukimura",
        "Mōri Motonari",
        "Hōjō Ujiyasu",
        "Akechi Mitsuhide",
    ],
}


# ==========================================
# Dataset: Mixed (Military / Scholars / Cities)
# ==========================================

MIXED_DATASET: Dict[str, List[str]] = {
    "mil": [
        "Oda Nobunaga",           # Japan, 16th century
        "Cao Cao",                # China, 3rd century
        "Napoleon",               # France, 19th century
        "Hannibal",               # Carthage, 3rd century BC
        "Alexander the Great",    # Macedonia, 4th century BC
    ],
    "sch": [
        "Albert Einstein",        # Physics
        "Abraham Maslow",         # Psychology
        "René Descartes",         # Philosophy
        "Jean-Henri Fabre",       # Biology (entomology)
        "Paracelsus",             # Toxicology/Medicine
    ],
    "city": [
        "Tokyo",
        "London",
        "New York City",
        "Paris",
        "Berlin",
    ],
}


# ==========================================
# Registry
# ==========================================

DATASETS: Dict[str, Dict[str, List[str]]] = {
    "warlords": SENGOKU_WARLORDS,
    "mixed": MIXED_DATASET,
}


def get_dataset(name: str) -> Dict[str, List[str]]:
    """Get dataset definition by name."""
    if name not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(f"Unknown dataset: '{name}'. Available: {available}")
    return DATASETS[name]


def make_article_id(prefix: str, title: str) -> str:
    """
    Create article ID from prefix and title.
    
    Examples:
        make_article_id("mil", "Oda Nobunaga") -> "mil_oda_nobunaga"
        make_article_id("sch", "René Descartes") -> "sch_rene_descartes"
    """
    normalized = title.lower()
    normalized = normalized.replace(" ", "_")
    # Normalize Unicode characters
    replacements = {
        "ō": "o", "ū": "u",
        "é": "e", "è": "e", "ê": "e",
        "ä": "a", "ö": "o", "ü": "u",
        "ñ": "n",
    }
    for src, dst in replacements.items():
        normalized = normalized.replace(src, dst)
    
    return f"{prefix}_{normalized}"
