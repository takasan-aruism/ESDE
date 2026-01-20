"""
ESDE Engine v5.3.3 - Configuration (Audit Patch)

SINGLE SOURCE OF TRUTH for all thresholds and constants.
All modules MUST import from here.
"""
from typing import Set

VERSION = "5.3.3"

# =============================================================================
# Phase 7B+ Constants (CENTRALIZED - Single Source of Truth)
# =============================================================================

# Competing hypothesis threshold
# A hypothesis is "competing" if score >= COMPETE_TH
COMPETE_TH = 0.15

# Minimum score for Route A when any plausible typo exists
# Even without explicit typo candidates, A should not be 0
# if the token could plausibly be a typo (short word, near dictionary word)
ROUTE_A_MIN_SCORE = 0.15

# Volatility thresholds (for status determination)
VOL_LOW_TH = 0.25    # Below = candidate
VOL_HIGH_TH = 0.50   # Above = quarantine

# Legacy aliases (for backward compatibility)
DEFAULT_LOW_VOLATILITY_THRESHOLD = VOL_LOW_TH
DEFAULT_HIGH_VOLATILITY_THRESHOLD = VOL_HIGH_TH
DEFAULT_CONFIDENCE_FLOOR = 0.40

# =============================================================================
# File paths
# =============================================================================

# File paths
SYNAPSE_FILE = "esde_synapses_v3.json"
GLOSSARY_FILE = "glossary_results.json"

# Phase 7A: Unknown Queue settings
QUEUE_FILE_PATH = "./data/unknown_queue.jsonl"
QUEUE_INCLUDE_NOISE = False
QUEUE_BUFFER_SIZE = 10

# Phase 7A+: Unknown Routing Variance Gate
UNKNOWN_MARGIN_TH = 0.20
UNKNOWN_ENTROPY_TH = 0.90

# Phase 7A+: Context feature weights
CONTEXT_TITLE_LIKE_BOOST = 0.20
CONTEXT_CAPITALIZED_BOOST = 0.20
CONTEXT_QUOTE_BOOST = 0.20
CONTEXT_TYPO_PENALTY_TITLE = 0.15

# Processing limits
MAX_SYNSETS_PER_TOKEN = 3
ALLOWED_POS = {'n', 'v', 'a', 'r', 's'}  # Added 's' for Satellite Adjective 2026/01/11

# Gate settings
TOP_CONCEPTS = 5
AXIS_TOP_LEVELS = 1

# Safety
DEDUP_PER_TOKEN_CONCEPT = True

# Phase 6A: Proximity expansion
ENABLE_PROXIMITY_EXPANSION = True
MAX_PROXY_DEPTH = 1
DEBUG_PROXIMITY = False

# Decay values
DECAY_SIMILAR = 0.8
DECAY_HYPERNYM = 0.5
DECAY_DERIVATIONAL = 0.6
DECAY_INSTANCE_HYPERNYM = 0.6
DECAY_HOLONYM = 0.4

# Variance detection (per-synset)
VARIANCE_MARGIN_THRESHOLD = 0.25
VARIANCE_ENTROPY_THRESHOLD = 0.85
VARIANCE_DOWNWEIGHT = 0.6

# Dynamic threshold
MIN_SCORE_THRESHOLD = 0.3
DYNAMIC_FLOOR_RATIO = 0.55

# Minimum token length
MIN_TOKEN_LENGTH = 2

# Stopwords - SKIP ENTIRELY
STOPWORDS: Set[str] = {
    # Pronouns
    'i', 'you', 'your', 'yours', 'he', 'she', 'they', 'we', 'me', 'him', 'her', 'us', 'them',
    'my', 'mine', 'his', 'hers', 'its', 'our', 'ours', 'their', 'theirs',
    'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'themselves',
    # Be verbs
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    # Do/Have verbs
    'do', 'did', 'does', 'done', 'doing',
    'have', 'has', 'had', 'having',
    # Modal verbs
    'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might', 'must',
    # Articles & determiners
    'a', 'an', 'the', 'this', 'that', 'these', 'those', 'some', 'any', 'no', 'every',
    # Prepositions
    'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'into', 'onto', 'upon',
    'about', 'after', 'before', 'between', 'under', 'over', 'through', 'during',
    # Conjunctions
    'and', 'or', 'but', 'nor', 'so', 'yet', 'both', 'either', 'neither',
    # Common adverbs
    'again', 'also', 'already', 'always', 'never', 'ever', 'just', 'only',
    'very', 'too', 'quite', 'rather', 'really', 'still', 'even', 'almost',
    'now', 'then', 'here', 'there', 'when', 'where', 'how', 'why',
    # Question words
    'what', 'which', 'who', 'whom', 'whose',
    # Others
    'it', 'as', 'if', 'not', 'than', 'such', 'like'
}

# Typo detection
TYPO_MAX_EDIT_DISTANCE = 2
TYPO_MIN_CONFIDENCE = 0.7
TYPO_REFERENCE_WORDS: Set[str] = {
    'oops', 'hello', 'love', 'like', 'make', 'take', 'give', 'have',
    'come', 'work', 'play', 'help', 'need', 'want', 'know', 'think',
    'feel', 'look', 'find', 'tell', 'call', 'leave', 'keep', 'begin',
    'seem', 'show', 'hear', 'turn', 'start', 'might', 'should', 'could'
}

# Proper noun detection patterns
PROPER_NOUN_INDICATORS: Set[str] = {
    'demigod', 'deity', 'god', 'goddess', 'hero', 'character',
    'person', 'celebrity', 'figure', 'entity', 'being'
}

# Title-like phrases for context detection
TITLE_LIKE_PHRASES: Set[str] = {
    # Song titles / famous phrases
    'did it again', 'i did it again', 'oops i did it again',
    'love you', 'i love you', 'love me', 'love song',
    'let it be', 'let it go', 'let me be',
    'hit me baby', 'baby one more time',
    'come as you are', 'as you are',
    'we are the champions', 'we are',
    'born this way', 'this way',
    'call me maybe', 'call me',
    'shake it off', 'shake it',
    'rolling in the deep', 'in the deep',
    'all you need is love', 'need is love',
    'hey jude', 'hey ya', 'hey there',
    'gonna give you up', 'never gonna',
    'beat it', 'just beat it',
    'thriller', 'billie jean',
    'bohemian rhapsody', 'we will rock you',
    # Common expressions/titles
    'ops center', 'ops team', 'dev ops', 'devops',
    'once upon a time', 'upon a time',
    'to be or not to be', 'to be or not',
    'the end', 'the beginning',
    'part one', 'part two', 'part 1', 'part 2',
    'chapter one', 'chapter two',
    'act one', 'act two',
    # Tech/gaming terms
    'git ops', 'sec ops', 'ml ops', 'mlops',
    'no cap', 'on god', 'fr fr', 'ngl',
}
# =============================================================================
# Phase 8: Sensor V2 Configuration
# =============================================================================
SENSOR_TOP_K = 5
SENSOR_MAX_SYNSETS_PER_TOKEN = 3
STRICT_SYNAPSE_ONLY = False
SENSOR_ALLOWED_POS = {'n', 'v', 'a', 'r', 's'}