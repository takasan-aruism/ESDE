#!/usr/bin/env python3
"""
ESDE Lexicon v2 — WordNet-based Candidate Supply + LLM Mapping
==============================================================

Pipeline:
  1. SEED     — Human/audit-defined synsets with sense boundaries
  2. EXPAND   — WordNet relation traversal (controlled depth/width)
  3. CLEAN    — Dedup, POS normalization, boundary filtering
  4. MAP      — LLM classifies each candidate into ESDE slots (no generation)
  5. AUDIT    — Status: Proposed → Audited → Core

Usage:
  python3 wn_lexicon.py EMO.like          # Full pipeline
  python3 wn_lexicon.py EMO.like --seed   # Seed + Expand only (no LLM)
  python3 wn_lexicon.py EMO.like --dry    # Show prompt only
"""

import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# ════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════

EXPAND_CONFIG = {
    # relation_type: {"max_terms": N, "pos_filter": [pos...] or None, "depth": 1}
    "synonyms":                  {"max_terms": 30, "pos_filter": None, "depth": 1},
    "hyponyms":                  {"max_terms": 20, "pos_filter": None, "depth": 1},
    "hypernyms":                 {"max_terms": 5,  "pos_filter": None, "depth": 1},
    "derivationally_related":    {"max_terms": 30, "pos_filter": None, "depth": 1},
    "similar_to":                {"max_terms": 15, "pos_filter": ["adj", "adv"], "depth": 1},
    "also_see":                  {"max_terms": 10, "pos_filter": ["adj"], "depth": 1},
    "antonyms":                  {"max_terms": 10, "pos_filter": None, "depth": 1},
}

# ESDE 20 slots (4 axes × levels)
SLOTS = [
    "resonance.superficial", "resonance.structural",
    "resonance.essential", "resonance.existential",
    "symmetry.destructive", "symmetry.inclusive",
    "symmetry.transformative", "symmetry.generative", "symmetry.cyclical",
    "lawfulness.predictable", "lawfulness.emergent",
    "lawfulness.contingent", "lawfulness.necessary",
    "experience.discovery", "experience.creation", "experience.comprehension",
    "value_generation.functional", "value_generation.aesthetic",
    "value_generation.ethical", "value_generation.sacred",
]

NA_REASONS = [
    "axis_generic",           # Word describes the axis, not the atom
    "out_of_atom_scope",      # Beyond sense boundary
    "too_metaphorical",       # Connection too indirect
    "unclear_level_definition", # Slot definition ambiguous
    "duplicate_semantics",    # Another word already covers this meaning
]


# ════════════════════════════════════════════════════════════════
#  SEED DEFINITIONS
# ════════════════════════════════════════════════════════════════

@dataclass
class SeedSynset:
    synset_id: str
    lemma: str
    pos: str
    definition: str
    in_scope: str       # What this sense covers for the atom
    out_of_scope: str   # Explicit exclusions (sense boundary)

@dataclass
class AtomSeed:
    atom_id: str
    category: str       # EMO, ACT, META, etc.
    kanji: str
    kanji_field: str    # Semantic field of the kanji
    boundary: str       # Overall sense boundary description
    seeds: list         # List of SeedSynset
    exclude_lemmas: list = field(default_factory=list)  # Hard exclusions


# ── EMO.like seed definition ──────────────────────────────────
ATOM_SEEDS = {
    "EMO.like": AtomSeed(
        atom_id="EMO.like",
        category="EMO",
        kanji="好",
        kanji_field="favorable regard, fondness, affinity, to be fond of, pleasing",
        boundary=(
            "Covers: positive emotional regard, fondness, affection, enjoyment of, "
            "warmth toward, favorable disposition. "
            "Excludes: rational preference/choice (prefer X over Y), "
            "similarity (like = similar to), desire/lust (→ EMO.desire), "
            "deep love/devotion (→ EMO.love)"
        ),
        seeds=[
            SeedSynset(
                synset_id="like.v.02",
                lemma="like",
                pos="v",
                definition="find enjoyable or agreeable",
                in_scope="Core verb: finding something enjoyable",
                out_of_scope="'would like to' (wish), 'like' (similar)"
            ),
            SeedSynset(
                synset_id="fondness.n.01",
                lemma="fondness",
                pos="n",
                definition="a positive feeling of liking",
                in_scope="Core noun: the state of liking",
                out_of_scope="fondness as weakness/vulnerability"
            ),
            SeedSynset(
                synset_id="affection.n.01",
                lemma="affection",
                pos="n",
                definition="a positive feeling of liking",
                in_scope="Warm positive regard",
                out_of_scope="Deep romantic love (→ EMO.love)"
            ),
            SeedSynset(
                synset_id="likable.a.01",
                lemma="likable",
                pos="adj",
                definition="easy to like; agreeable",
                in_scope="Quality of evoking liking",
                out_of_scope=""
            ),
        ],
        exclude_lemmas=["similar", "alike", "such_as", "prefer", "desire", "love"]
    ),

    "EMO.anger": AtomSeed(
        atom_id="EMO.anger",
        category="EMO",
        kanji="怒",
        kanji_field="anger, wrath, rage, indignation, to be angry",
        boundary=(
            "Covers: emotional state of anger from irritation to rage, "
            "indignation, fury, wrath, ire. "
            "Excludes: hatred as sustained disposition (→ EMO.hate), "
            "contempt as looking down (→ EMO.contempt), "
            "fear-based aggression (→ EMO.fear)"
        ),
        seeds=[
            SeedSynset(
                synset_id="anger.n.01",
                lemma="anger",
                pos="n",
                definition="a strong emotion; a feeling of intense displeasure",
                in_scope="Core emotion of anger",
                out_of_scope="anger as sin (theological)"
            ),
            SeedSynset(
                synset_id="anger.n.02",
                lemma="anger",
                pos="n",
                definition="the state of being angry",
                in_scope="State of being angry",
                out_of_scope=""
            ),
            SeedSynset(
                synset_id="angry.a.01",
                lemma="angry",
                pos="adj",
                definition="feeling or showing anger",
                in_scope="Adjectival form",
                out_of_scope="angry (of wounds/weather)"
            ),
            SeedSynset(
                synset_id="rage.n.01",
                lemma="rage",
                pos="n",
                definition="a feeling of intense anger",
                in_scope="Intense end of anger spectrum",
                out_of_scope="rage as fashion/trend"
            ),
        ],
        exclude_lemmas=["hate", "hatred", "contempt", "fear"]
    ),
}


# ════════════════════════════════════════════════════════════════
#  STEP 2: EXPAND (WordNet traversal)
# ════════════════════════════════════════════════════════════════

@dataclass
class Candidate:
    lemma: str
    pos: str
    source_relation: str
    source_synset: str
    definition: str = ""

def try_import_wordnet():
    """Try to import WordNet from available packages."""
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets('test')
        return wn, 'nltk'
    except:
        pass
    try:
        import wn as wn_pkg
        # Check if data is loaded
        wn_pkg.synsets('test')
        return wn_pkg, 'wn'
    except:
        pass
    return None, None

def expand_nltk(seed: AtomSeed) -> list[Candidate]:
    """Expand seeds using NLTK WordNet."""
    from nltk.corpus import wordnet as wn
    
    candidates = []
    seen = set()
    
    for s in seed.seeds:
        synset = wn.synset(s.synset_id)
        
        # Synonyms (lemmas in same synset)
        cfg = EXPAND_CONFIG["synonyms"]
        for lemma in synset.lemmas()[:cfg["max_terms"]]:
            name = lemma.name().replace('_', ' ')
            key = (name.lower(), _nltk_pos(synset.pos()))
            if key not in seen:
                seen.add(key)
                candidates.append(Candidate(
                    lemma=name, pos=_nltk_pos(synset.pos()),
                    source_relation="synonym",
                    source_synset=s.synset_id,
                    definition=synset.definition()
                ))
        
        # Hyponyms (1 level)
        cfg = EXPAND_CONFIG["hyponyms"]
        count = 0
        for hypo in synset.hyponyms():
            if count >= cfg["max_terms"]:
                break
            for lemma in hypo.lemmas():
                name = lemma.name().replace('_', ' ')
                key = (name.lower(), _nltk_pos(hypo.pos()))
                if key not in seen:
                    seen.add(key)
                    candidates.append(Candidate(
                        lemma=name, pos=_nltk_pos(hypo.pos()),
                        source_relation="hyponym",
                        source_synset=synset.name(),
                        definition=hypo.definition()
                    ))
                    count += 1
        
        # Hypernyms (1 level, limited)
        cfg = EXPAND_CONFIG["hypernyms"]
        count = 0
        for hyper in synset.hypernyms():
            if count >= cfg["max_terms"]:
                break
            for lemma in hyper.lemmas():
                name = lemma.name().replace('_', ' ')
                key = (name.lower(), _nltk_pos(hyper.pos()))
                if key not in seen:
                    seen.add(key)
                    candidates.append(Candidate(
                        lemma=name, pos=_nltk_pos(hyper.pos()),
                        source_relation="hypernym",
                        source_synset=synset.name(),
                        definition=hyper.definition()
                    ))
                    count += 1
        
        # Derivationally related forms
        cfg = EXPAND_CONFIG["derivationally_related"]
        count = 0
        for lemma_obj in synset.lemmas():
            for related in lemma_obj.derivationally_related_forms():
                if count >= cfg["max_terms"]:
                    break
                name = related.name().replace('_', ' ')
                key = (name.lower(), _nltk_pos(related.synset().pos()))
                if key not in seen:
                    seen.add(key)
                    candidates.append(Candidate(
                        lemma=name, pos=_nltk_pos(related.synset().pos()),
                        source_relation="derivational",
                        source_synset=synset.name(),
                        definition=related.synset().definition()
                    ))
                    count += 1
        
        # Similar_to (adj only)
        cfg = EXPAND_CONFIG["similar_to"]
        count = 0
        for sim in synset.similar_tos():
            pos = _nltk_pos(sim.pos())
            if cfg["pos_filter"] and pos not in cfg["pos_filter"]:
                continue
            if count >= cfg["max_terms"]:
                break
            for lemma in sim.lemmas():
                name = lemma.name().replace('_', ' ')
                key = (name.lower(), pos)
                if key not in seen:
                    seen.add(key)
                    candidates.append(Candidate(
                        lemma=name, pos=pos,
                        source_relation="similar_to",
                        source_synset=synset.name(),
                        definition=sim.definition()
                    ))
                    count += 1
        
        # Also_see (adj only)
        cfg = EXPAND_CONFIG["also_see"]
        count = 0
        for also in synset.also_sees():
            pos = _nltk_pos(also.pos())
            if cfg["pos_filter"] and pos not in cfg["pos_filter"]:
                continue
            if count >= cfg["max_terms"]:
                break
            for lemma in also.lemmas():
                name = lemma.name().replace('_', ' ')
                key = (name.lower(), pos)
                if key not in seen:
                    seen.add(key)
                    candidates.append(Candidate(
                        lemma=name, pos=pos,
                        source_relation="also_see",
                        source_synset=synset.name(),
                        definition=also.definition()
                    ))
                    count += 1
        
        # Antonyms
        cfg = EXPAND_CONFIG["antonyms"]
        count = 0
        for lemma_obj in synset.lemmas():
            for ant in lemma_obj.antonyms():
                if count >= cfg["max_terms"]:
                    break
                name = ant.name().replace('_', ' ')
                key = (name.lower(), _nltk_pos(ant.synset().pos()))
                if key not in seen:
                    seen.add(key)
                    candidates.append(Candidate(
                        lemma=name, pos=_nltk_pos(ant.synset().pos()),
                        source_relation="antonym",
                        source_synset=synset.name(),
                        definition=ant.synset().definition()
                    ))
                    count += 1
    
    return candidates


def _nltk_pos(pos_tag: str) -> str:
    """Convert NLTK POS to ESDE POS."""
    return {"n": "n", "v": "v", "a": "adj", "s": "adj", "r": "adv"}.get(pos_tag, pos_tag)


# ════════════════════════════════════════════════════════════════
#  EMBEDDED FALLBACK DATA (for environments without WordNet)
# ════════════════════════════════════════════════════════════════

EMBEDDED_CANDIDATES = {
    "EMO.like": [
        # ── From like.v.02 "find enjoyable or agreeable" ──
        # Synonyms
        Candidate("like", "v", "synonym", "like.v.02", "find enjoyable or agreeable"),
        Candidate("enjoy", "v", "synonym", "like.v.02", "find enjoyable or agreeable"),
        # Hyponyms of like.v.02
        Candidate("fancy", "v", "hyponym", "like.v.02", "have a fancy or particular liking for"),
        Candidate("cotton to", "v", "hyponym", "like.v.02", "take a liking to"),
        Candidate("take to", "v", "hyponym", "like.v.02", "develop a habit; begin to like"),
        Candidate("savor", "v", "hyponym", "like.v.02", "derive or receive pleasure from"),
        Candidate("relish", "v", "hyponym", "like.v.02", "derive or receive pleasure from"),
        Candidate("bask", "v", "hyponym", "like.v.02", "derive enjoyment from"),
        Candidate("delight", "v", "hyponym", "like.v.02", "take delight in"),
        Candidate("revel", "v", "hyponym", "like.v.02", "take delight in"),
        # Derivational from like
        Candidate("liking", "n", "derivational", "like.v.02", "a feeling of pleasure and enjoyment"),
        Candidate("likable", "adj", "derivational", "like.v.02", "easy to like; agreeable"),
        Candidate("likeable", "adj", "derivational", "like.v.02", "easy to like; agreeable"),
        # Antonym
        Candidate("dislike", "v", "antonym", "like.v.02", "have or feel a dislike or distaste for"),
        
        # ── From fondness.n.01 "a positive feeling of liking" ──
        Candidate("fondness", "n", "synonym", "fondness.n.01", "a positive feeling of liking"),
        Candidate("affection", "n", "synonym", "fondness.n.01", "a positive feeling of liking"),
        Candidate("heart", "n", "synonym", "fondness.n.01", "a positive feeling of liking"),
        Candidate("warmness", "n", "synonym", "fondness.n.01", "a positive feeling of liking"),
        Candidate("warmheartedness", "n", "synonym", "fondness.n.01", "a positive feeling of liking"),
        Candidate("philia", "n", "synonym", "fondness.n.01", "a positive feeling of liking"),
        # Hyponyms of fondness
        Candidate("attachment", "n", "hyponym", "fondness.n.01", "a feeling of affection for a person or place"),
        Candidate("regard", "n", "hyponym", "fondness.n.01", "a feeling of affection for a person"),
        Candidate("respect", "n", "hyponym", "fondness.n.01", "a feeling of regard for"),
        Candidate("soft spot", "n", "hyponym", "fondness.n.01", "a sentimental fondness"),
        Candidate("tenderness", "n", "hyponym", "fondness.n.01", "a tendency to express warm and affectionate feeling"),
        Candidate("protectiveness", "n", "hyponym", "fondness.n.01", "a feeling of protective affection"),
        # Hypernym of fondness
        Candidate("feeling", "n", "hypernym", "fondness.n.01", "the experiencing of affective and emotional states"),
        # Derivational from fond
        Candidate("fond", "adj", "derivational", "fondness.n.01", "having or displaying warmth"),
        Candidate("fondly", "adv", "derivational", "fondness.n.01", "with fondness; with love"),
        
        # ── From affection.n.01 ──
        Candidate("affectionate", "adj", "derivational", "affection.n.01", "having or displaying warmth"),
        Candidate("affectionateness", "n", "derivational", "affection.n.01", "a quality proceeding from feelings of affection"),
        # Hyponyms
        Candidate("warmth", "n", "hyponym", "affection.n.01", "a quality proceeding from feelings of affection"),
        Candidate("friendliness", "n", "hyponym", "affection.n.01", "a friendly disposition"),
        Candidate("kindliness", "n", "hyponym", "affection.n.01", "a kind and sympathetic disposition"),
        
        # ── From likable.a.01 ──
        Candidate("appealing", "adj", "similar_to", "likable.a.01", "able to attract interest or draw favorable attention"),
        Candidate("sympathetic", "adj", "similar_to", "likable.a.01", "of characters in literature or drama that evoke sympathy"),
        Candidate("endearing", "adj", "also_see", "likable.a.01", "lovable especially in a childlike or naive way"),
        Candidate("lovable", "adj", "also_see", "likable.a.01", "having characteristics that attract love or affection"),
        
        # ── Extended from enjoy.v.01 (synonym chain) ──
        Candidate("enjoyment", "n", "derivational", "enjoy.v.01", "the pleasure felt when having a good time"),
        Candidate("enjoyable", "adj", "derivational", "enjoy.v.01", "affording satisfaction or pleasure"),
        Candidate("pleasant", "adj", "similar_to", "enjoy.v.01", "affording pleasure; being in harmony with your taste"),
        Candidate("agreeable", "adj", "similar_to", "enjoy.v.01", "conforming to your own liking or feelings"),
        Candidate("pleasurable", "adj", "similar_to", "enjoy.v.01", "affording satisfaction or pleasure"),
        
        # ── From delight (hyponym chain) ──
        Candidate("delight", "n", "derivational", "delight.v.01", "a feeling of extreme pleasure or satisfaction"),
        Candidate("delightful", "adj", "derivational", "delight.v.01", "greatly pleasing or entertaining"),
        Candidate("charming", "adj", "similar_to", "delight.v.01", "pleasing or delighting"),
        Candidate("enchanting", "adj", "similar_to", "delight.v.01", "capturing interest as if by a spell"),
        Candidate("captivating", "adj", "similar_to", "delight.v.01", "capturing interest as if by a spell"),
        
        # ── Additional related forms ──
        Candidate("favor", "n", "hyponym", "fondness.n.01", "an inclination to approve"),
        Candidate("favorable", "adj", "derivational", "favor.n.01", "encouraging or approving"),
        Candidate("preference", "n", "hyponym", "fondness.n.01", "a predisposition in favor of something"),
        Candidate("predilection", "n", "hyponym", "fondness.n.01", "a predisposition in favor of something"),
        Candidate("inclination", "n", "hyponym", "fondness.n.01", "an attitude of mind especially one that favors"),
        Candidate("partiality", "n", "hyponym", "fondness.n.01", "a predisposition to like something"),
        Candidate("appreciation", "n", "hyponym", "fondness.n.01", "understanding of the nature or meaning"),
        Candidate("admiration", "n", "hyponym", "fondness.n.01", "a feeling of delighted approval and liking"),
        Candidate("adoration", "n", "hyponym", "fondness.n.01", "a feeling of profound love and admiration"),
        
        # ── Antonyms / contrast ──
        Candidate("dislike", "n", "antonym", "fondness.n.01", "a feeling of aversion or antipathy"),
        Candidate("aversion", "n", "antonym", "fondness.n.01", "a feeling of intense dislike"),
        Candidate("antipathy", "n", "antonym", "fondness.n.01", "a feeling of intense dislike"),
        Candidate("distaste", "n", "antonym", "fondness.n.01", "a feeling of intense dislike"),
        
        # ── Verbs from derivational chains ──
        Candidate("cherish", "v", "hyponym", "like.v.02", "be fond of; be attached to"),
        Candidate("treasure", "v", "hyponym", "like.v.02", "hold dear"),
        Candidate("appreciate", "v", "hyponym", "like.v.02", "recognize with gratitude; be grateful for"),
        Candidate("admire", "v", "hyponym", "like.v.02", "feel admiration for"),
        Candidate("adore", "v", "hyponym", "like.v.02", "love intensely"),
        Candidate("favor", "v", "hyponym", "like.v.02", "treat gently or carefully"),
        Candidate("warm to", "v", "hyponym", "like.v.02", "become affectionate or kind"),
    ],

    "EMO.anger": [
        # ── From anger.n.01 "a strong emotion" ──
        Candidate("anger", "n", "synonym", "anger.n.01", "a strong emotion; a feeling of intense displeasure"),
        Candidate("choler", "n", "synonym", "anger.n.01", "a strong emotion; a feeling of intense displeasure"),
        Candidate("ire", "n", "synonym", "anger.n.01", "a strong emotion; a feeling of intense displeasure"),
        # Hyponyms
        Candidate("fury", "n", "hyponym", "anger.n.01", "a feeling of intense anger"),
        Candidate("rage", "n", "hyponym", "anger.n.01", "a feeling of intense anger"),
        Candidate("wrath", "n", "hyponym", "anger.n.01", "intense anger"),
        Candidate("indignation", "n", "hyponym", "anger.n.01", "a feeling of righteous anger"),
        Candidate("outrage", "n", "hyponym", "anger.n.01", "a feeling of righteous anger"),
        Candidate("umbrage", "n", "hyponym", "anger.n.01", "a feeling of anger caused by being offended"),
        Candidate("offense", "n", "hyponym", "anger.n.01", "a feeling of anger caused by being offended"),
        Candidate("huffiness", "n", "hyponym", "anger.n.01", "an angry disturbance"),
        Candidate("dander", "n", "hyponym", "anger.n.01", "a feeling of anger and animosity"),
        Candidate("dudgeon", "n", "hyponym", "anger.n.01", "a feeling of intense indignation"),
        Candidate("infuriation", "n", "hyponym", "anger.n.01", "a feeling of intense irritation"),
        # Hypernym
        Candidate("emotion", "n", "hypernym", "anger.n.01", "any strong feeling"),
        Candidate("displeasure", "n", "hypernym", "anger.n.01", "the feeling of being displeased"),
        # Derivational
        Candidate("angry", "adj", "derivational", "anger.n.01", "feeling or showing anger"),
        Candidate("angered", "adj", "derivational", "anger.n.01", "marked by extreme anger"),
        Candidate("angrily", "adv", "derivational", "anger.n.01", "with anger"),
        Candidate("anger", "v", "derivational", "anger.n.01", "make angry"),
        Candidate("enrage", "v", "derivational", "rage.n.01", "put into a rage; make furious"),
        Candidate("infuriate", "v", "derivational", "anger.n.01", "make furious"),
        
        # ── From rage.n.01 ──
        Candidate("fury", "n", "synonym", "rage.n.01", "a feeling of intense anger"),
        Candidate("madness", "n", "synonym", "rage.n.01", "a feeling of intense anger"),
        Candidate("furious", "adj", "derivational", "rage.n.01", "marked by extreme anger"),
        Candidate("livid", "adj", "similar_to", "rage.n.01", "furiously angry"),
        Candidate("seething", "adj", "similar_to", "rage.n.01", "in a state of agitation"),
        Candidate("incensed", "adj", "similar_to", "rage.n.01", "angered at something unjust"),
        
        # ── From angry.a.01 ──
        Candidate("irate", "adj", "similar_to", "angry.a.01", "feeling or showing extreme anger"),
        Candidate("wrathful", "adj", "similar_to", "angry.a.01", "vehemently incensed and condemnatory"),
        Candidate("indignant", "adj", "similar_to", "angry.a.01", "angered at something unjust"),
        Candidate("outraged", "adj", "similar_to", "angry.a.01", "angered at something unjust"),
        Candidate("exasperated", "adj", "similar_to", "angry.a.01", "greatly annoyed; out of patience"),
        Candidate("irritated", "adj", "similar_to", "angry.a.01", "aroused to impatience or anger"),
        Candidate("annoyed", "adj", "similar_to", "angry.a.01", "aroused to impatience or anger"),
        Candidate("vexed", "adj", "similar_to", "angry.a.01", "troubled persistently"),
        Candidate("provoked", "adj", "similar_to", "angry.a.01", "incited, especially deliberately, to anger"),
        Candidate("resentful", "adj", "similar_to", "angry.a.01", "full of or marked by resentment"),
        Candidate("bitter", "adj", "similar_to", "angry.a.01", "marked by strong resentment or cynicism"),
        # Antonym
        Candidate("calm", "adj", "antonym", "angry.a.01", "not agitated; without losing self-possession"),
        
        # ── Lower-intensity anger (from irritation chain) ──
        Candidate("irritation", "n", "hyponym", "anger.n.01", "the psychological state of being irritated"),
        Candidate("annoyance", "n", "hyponym", "anger.n.01", "the psychological state of being annoyed"),
        Candidate("vexation", "n", "hyponym", "anger.n.01", "the psychological state of being vexed"),
        Candidate("exasperation", "n", "hyponym", "anger.n.01", "an exasperated feeling of annoyance"),
        Candidate("frustration", "n", "hyponym", "anger.n.01", "a feeling of annoyance from being hindered"),
        Candidate("aggravation", "n", "hyponym", "anger.n.01", "an exasperated feeling of annoyance"),
        Candidate("resentment", "n", "hyponym", "anger.n.01", "a feeling of deep and bitter anger and ill-will"),
        Candidate("bitterness", "n", "hyponym", "anger.n.01", "a feeling of deep and bitter anger"),
        Candidate("rancor", "n", "hyponym", "anger.n.01", "a feeling of deep and bitter anger"),
        Candidate("animosity", "n", "hyponym", "anger.n.01", "a feeling of ill will arousing active hostility"),
        
        # ── Verbs ──
        Candidate("irritate", "v", "hyponym", "anger.v.01", "cause annoyance in"),
        Candidate("annoy", "v", "hyponym", "anger.v.01", "cause annoyance in"),
        Candidate("vex", "v", "hyponym", "anger.v.01", "cause annoyance in"),
        Candidate("provoke", "v", "hyponym", "anger.v.01", "provide the needed stimulus for"),
        Candidate("incense", "v", "hyponym", "anger.v.01", "make furious"),
        Candidate("aggravate", "v", "hyponym", "anger.v.01", "make worse"),
        Candidate("gall", "v", "hyponym", "anger.v.01", "irritate or vex"),
        Candidate("rankle", "v", "hyponym", "anger.v.01", "gnaw into; make resentful or angry"),
        Candidate("fume", "v", "hyponym", "anger.v.01", "be mad, angry, or furious"),
        Candidate("seethe", "v", "hyponym", "anger.v.01", "be in an agitated emotional state"),
        Candidate("bristle", "v", "hyponym", "anger.v.01", "react in an offended or angry manner"),
        Candidate("flare up", "v", "hyponym", "anger.v.01", "erupt or intensify suddenly"),
    ],
}


# ════════════════════════════════════════════════════════════════
#  STEP 3: CLEAN (dedup, boundary filter, normalize)
# ════════════════════════════════════════════════════════════════

def clean_candidates(candidates: list[Candidate], seed: AtomSeed) -> list[Candidate]:
    """Remove duplicates, apply boundary exclusions, normalize."""
    seen = set()
    cleaned = []
    
    for c in candidates:
        # Normalize
        lemma_key = c.lemma.lower().strip()
        
        # Skip excluded lemmas
        if lemma_key in [x.lower() for x in seed.exclude_lemmas]:
            continue
        
        # Dedup by (lemma, pos)
        key = (lemma_key, c.pos)
        if key in seen:
            continue
        seen.add(key)
        
        cleaned.append(c)
    
    return cleaned


# ════════════════════════════════════════════════════════════════
#  STEP 4: MAP (LLM prompt generation)
# ════════════════════════════════════════════════════════════════

def build_slot_definitions() -> str:
    """Build detailed slot definitions for the mapper prompt."""
    return """
## Axis: Resonance (depth of engagement)
- resonance.superficial: Surface-level, casual, fleeting manifestation. First impression.
- resonance.structural: Pattern-level, habitual, systematic. Reliable recurring form.
- resonance.essential: Core identity, deeply personal, defining. Cannot be separated from self.
- resonance.existential: Transcends individual; universal, philosophical, boundary of existence.

## Axis: Symmetry (relational dynamics)
- symmetry.destructive: Dark side, excess, pathology. When this atom goes wrong.
- symmetry.inclusive: Absorbing, welcoming, encompassing. Drawing inward.
- symmetry.transformative: Changing form, catalyzing, metamorphosis. Becoming something else.
- symmetry.generative: Producing, creating, spawning new things. Outward generation.
- symmetry.cyclical: Recurring, rhythmic, seasonal. Patterns that repeat.

## Axis: Lawfulness (predictability of emergence)
- lawfulness.predictable: Expected, reliable, follows known patterns.
- lawfulness.emergent: Surprising, spontaneous, arising unexpectedly.
- lawfulness.contingent: Dependent on conditions, situational, could go either way.
- lawfulness.necessary: Inevitable, inherent, cannot NOT occur given the conditions.

## Axis: Experience (mode of knowing)
- experience.discovery: First encounter, revelation, the "aha" moment.
- experience.creation: Active making, crafting, building something from this atom.
- experience.comprehension: Deep understanding, integration into worldview.

## Axis: Value Generation (what kind of value is produced)
- value_generation.functional: Practical utility, serves a purpose, gets things done.
- value_generation.aesthetic: Beauty, elegance, artistic or sensory appreciation.
- value_generation.ethical: Moral dimension, right/wrong, duty, integrity.
- value_generation.sacred: Transcendent, spiritual, beyond ordinary value.
"""


def build_mapper_prompt(seed: AtomSeed, candidates: list[Candidate]) -> str:
    """Build the LLM mapping prompt."""
    
    # Format candidate list
    cand_lines = []
    for i, c in enumerate(candidates, 1):
        cand_lines.append(f"  {i:3d}. {c.lemma} ({c.pos}) — {c.definition}")
    cand_text = "\n".join(cand_lines)
    
    prompt = f"""You are an ESDE coordinate observer. You will classify words into semantic slots.

## Your Atom
- ID: {seed.atom_id}
- Kanji: {seed.kanji}
- Semantic field: {seed.kanji_field}
- Boundary: {seed.boundary}

## ESDE Slot Definitions
{build_slot_definitions()}

## Candidate Words
The following {len(candidates)} words were extracted from WordNet's neighborhood of "{seed.atom_id}".
Your task: assign each word to ONE or more ESDE slots, or mark it N/A.

{cand_text}

## STRICT RULES
1. **NO GENERATION**: Do not add any word that is not in the list above.
2. **NO AXIS-GENERIC**: If a word describes the AXIS definition rather than this specific atom,
   mark it N/A with reason "axis_generic". Examples: "conditional", "inherent", "routine".
3. **ATOM SPECIFICITY**: Each word must relate to {seed.atom_id} specifically.
   If it could equally apply to any emotion, mark N/A with reason "axis_generic".
4. **MULTI-SLOT OK**: A word may appear in multiple slots if genuinely applicable.
5. **ANTONYMS**: Place antonyms in "symmetry.destructive" (they represent the dark/opposite side).

## Output Format (JSON)
{{
  "atom": "{seed.atom_id}",
  "slots": {{
    "resonance.superficial": {{
      "words": [
        {{"w": "example", "pos": "n", "reason": "why this word belongs here"}},
      ]
    }},
    // ... all 20 slots ...
  }},
  "na": [
    {{"w": "example", "pos": "n", "na_reason": "axis_generic", "detail": "describes axis not atom"}}
  ]
}}

Respond ONLY with the JSON. No preamble, no markdown fences.
"""
    return prompt


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ESDE Lexicon v2 — WordNet pipeline")
    parser.add_argument("atom", help="Atom ID (e.g., EMO.like)")
    parser.add_argument("--seed", action="store_true", help="Seed + Expand only (no LLM)")
    parser.add_argument("--dry", action="store_true", help="Show prompt only")
    args = parser.parse_args()
    
    atom_id = args.atom
    if atom_id not in ATOM_SEEDS:
        print(f"❌ No seed definition for {atom_id}")
        print(f"   Available: {', '.join(ATOM_SEEDS.keys())}")
        sys.exit(1)
    
    seed = ATOM_SEEDS[atom_id]
    
    print(f"\n{'='*70}")
    print(f"  ESDE Lexicon v2 — {atom_id}")
    print(f"  Kanji: {seed.kanji} ({seed.kanji_field})")
    print(f"{'='*70}")
    
    # Step 1+2: Seed & Expand
    print(f"\n── Step 1: SEED ({len(seed.seeds)} synsets) ──")
    for s in seed.seeds:
        print(f"  {s.synset_id}: {s.lemma} ({s.pos}) — {s.definition}")
        print(f"    ✓ {s.in_scope}")
        if s.out_of_scope:
            print(f"    ✗ {s.out_of_scope}")
    
    print(f"\n── Step 2: EXPAND ──")
    wn, wn_type = try_import_wordnet()
    if wn and wn_type == 'nltk':
        print(f"  Using NLTK WordNet (live)")
        candidates = expand_nltk(seed)
    elif atom_id in EMBEDDED_CANDIDATES:
        print(f"  Using embedded candidate data (WordNet unavailable)")
        candidates = EMBEDDED_CANDIDATES[atom_id]
    else:
        print(f"  ❌ No WordNet and no embedded data for {atom_id}")
        sys.exit(1)
    
    # Step 3: Clean
    print(f"\n── Step 3: CLEAN ──")
    candidates = clean_candidates(candidates, seed)
    
    # Stats
    pos_counts = {}
    rel_counts = {}
    for c in candidates:
        pos_counts[c.pos] = pos_counts.get(c.pos, 0) + 1
        rel_counts[c.source_relation] = rel_counts.get(c.source_relation, 0) + 1
    
    print(f"  Candidates: {len(candidates)}")
    print(f"  POS: {pos_counts}")
    print(f"  Relations: {rel_counts}")
    print(f"  Excluded: {seed.exclude_lemmas}")
    
    print(f"\n  Full candidate list:")
    for i, c in enumerate(candidates, 1):
        print(f"    {i:3d}. {c.lemma:25s} {c.pos:4s}  [{c.source_relation:15s}] {c.definition[:60]}")
    
    # Save candidate list
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    
    cand_json = {
        "atom": atom_id,
        "seed_synsets": [asdict(s) for s in seed.seeds],
        "boundary": seed.boundary,
        "expand_config": EXPAND_CONFIG,
        "candidates": [asdict(c) for c in candidates],
        "stats": {"total": len(candidates), "pos": pos_counts, "relations": rel_counts},
    }
    cand_path = out_dir / f"{atom_id.replace('.','_')}_candidates.json"
    with open(cand_path, 'w') as f:
        json.dump(cand_json, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ Saved: {cand_path}")
    
    if args.seed:
        return
    
    # Step 4: Build prompt
    print(f"\n── Step 4: MAP (LLM prompt) ──")
    prompt = build_mapper_prompt(seed, candidates)
    
    if args.dry:
        print(prompt)
        prompt_path = out_dir / f"{atom_id.replace('.','_')}_prompt.txt"
        with open(prompt_path, 'w') as f:
            f.write(prompt)
        print(f"\n  ✅ Prompt saved: {prompt_path}")
        return
    
    print(f"  Prompt length: {len(prompt):,} chars")
    print(f"  (LLM call would go here — run locally with QwQ)")
    
    # Save prompt for local execution
    prompt_path = out_dir / f"{atom_id.replace('.','_')}_prompt.txt"
    with open(prompt_path, 'w') as f:
        f.write(prompt)
    print(f"  ✅ Prompt saved: {prompt_path}")


if __name__ == "__main__":
    main()
