"""
ESDE Phase 9: Token Feature Extractor
======================================
Generates 20-dimensional feature vectors for each token.

Philosophy: "Describe, but do not decide."

Feature Dimensions (20 total):

Part A: Structural & Contextual (10)
  0. pos_in_sentence_norm      - Relative position in sentence [0,1]
  1. sentence_index_norm       - Relative sentence position [0,1]
  2. section_index_norm        - Relative section position [0,1]
  3. is_capitalized            - First char is uppercase {0,1}
  4. is_year                   - Token is year 1000-2099 {0,1}
  5. distance_to_period_norm   - Distance to sentence end [0,1]
  6. is_be_aux                 - Be auxiliary verb {0,1}
  7. is_passive_participle     - Past participle in passive {0,1}
  8. inside_parentheses        - Inside () brackets {0,1}
  9. is_in_quote               - Inside quotation marks {0,1}

Part B: Psycholinguistic & Semantic (10)
  10. concreteness_score       - Brysbaert [0,1] or -1 (NULL)
  11. age_of_acquisition_norm  - Kuperman [0,1] or -1 (NULL)
  12. sensorimotor_intensity   - Lancaster [0,1] or -1 (NULL)
  13. emotional_valence        - Warriner [0,1] or -1 (NULL)
  14. is_proper_noun           - spaCy PROPN tag {0,1}
  15. is_action_verb           - VERB excluding aux {0,1}
  16. is_latin_suffix          - Has Latin abstract suffix {0,1}
  17. is_numeric_value         - Number (non-year) {0,1}
  18. dependency_head_dist     - spaCy head distance [0,1] or -1
  19. is_by_agent              - "by" in passive context {0,1}

Spec: Phase 9 W1 Feature Extraction v1.0
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple, Iterator
from pathlib import Path

# spaCy (optional but recommended)
try:
    import spacy
    from spacy.tokens import Token
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    Token = None

from .dict_provider import (
    DictionaryProvider,
    NULL_SCORE,
    DICT_CONCRETENESS,
    DICT_AOA,
    DICT_SENSORIMOTOR,
    DICT_VALENCE,
)


# ==========================================
# Constants
# ==========================================

FEATURE_EXTRACTOR_VERSION = "v1.0.0"
FEATURE_DIM = 20

# Feature names (for documentation and export)
FEATURE_NAMES = [
    # Part A: Structural
    "pos_in_sentence_norm",
    "sentence_index_norm",
    "section_index_norm",
    "is_capitalized",
    "is_year",
    "distance_to_period_norm",
    "is_be_aux",
    "is_passive_participle",
    "inside_parentheses",
    "is_in_quote",
    # Part B: Psycholinguistic
    "concreteness_score",
    "age_of_acquisition_norm",
    "sensorimotor_intensity",
    "emotional_valence",
    "is_proper_noun",
    "is_action_verb",
    "is_latin_suffix",
    "is_numeric_value",
    "dependency_head_dist",
    "is_by_agent",
]

# Be auxiliaries (for passive detection)
BE_AUXILIARIES = frozenset([
    "am", "is", "are", "was", "were", "be", "been", "being",
])

# Irregular past participles (common ones for passive detection)
IRREGULAR_PARTICIPLES = frozenset([
    "been", "born", "borne", "beaten", "become", "begun", "bent",
    "bitten", "bled", "blown", "broken", "brought", "built", "burnt",
    "bought", "caught", "chosen", "come", "cut", "dealt", "done",
    "drawn", "drunk", "driven", "eaten", "fallen", "fed", "felt",
    "fought", "found", "fled", "flown", "forbidden", "forgotten",
    "forgiven", "frozen", "given", "gone", "grown", "had", "heard",
    "held", "hidden", "hit", "hurt", "kept", "known", "laid", "led",
    "left", "lent", "let", "lit", "lost", "made", "meant", "met",
    "paid", "put", "read", "ridden", "risen", "run", "said", "seen",
    "sent", "set", "shaken", "shot", "shown", "shut", "slain", "slept",
    "spoken", "spent", "split", "spread", "stood", "stolen", "struck",
    "stuck", "sung", "sunk", "swept", "swum", "taken", "taught", "told",
    "thought", "thrown", "understood", "woken", "won", "worn", "written",
])

# Auxiliary verbs to exclude from "action verb"
AUX_VERBS = frozenset([
    "be", "am", "is", "are", "was", "were", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing", "done",
    "will", "would", "shall", "should",
    "can", "could", "may", "might", "must",
])

# Latin suffixes (abstract concept markers)
LATIN_SUFFIXES = (
    "tion", "sion", "ity", "ment", "ness", "ism",
    "ance", "ence", "acy", "dom", "ship", "hood",
)

# Number words
NUMBER_WORDS = frozenset([
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "billion", "trillion",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "half", "quarter", "dozen",
])

# Regex patterns
YEAR_PATTERN = re.compile(r'^(1\d{3}|20\d{2})$')  # 1000-2099
DIGIT_PATTERN = re.compile(r'\d')
ED_SUFFIX_PATTERN = re.compile(r'ed$', re.IGNORECASE)


# ==========================================
# Token Feature Record
# ==========================================

@dataclass
class TokenFeature:
    """
    Feature vector for a single token.
    """
    token: str
    lemma: str
    vector: Tuple[float, ...]  # 20-dim tuple
    
    # Position info (for traceability)
    token_idx: int = 0
    sentence_idx: int = 0
    char_start: int = 0
    char_end: int = 0
    
    # Section info (for pipeline, optional)
    section_idx: int = 0
    section_name: str = ""
    
    # Paragraph info (for structure statistics)
    paragraph_idx: int = 0          # Which paragraph this token is in
    paragraph_count: int = 1        # Total paragraphs in section/text
    sentences_in_paragraph: int = 1 # Sentences in this paragraph
    
    # Absolute values (分母) for structure statistics
    sentence_length: int = 0        # Tokens in this sentence
    total_sentences: int = 1        # Total sentences in section/text
    total_sections: int = 1         # Total sections in article
    
    def __post_init__(self):
        if isinstance(self.vector, list):
            object.__setattr__(self, 'vector', tuple(self.vector))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "token": self.token,
            "lemma": self.lemma,
            "vector": list(self.vector),
            "token_idx": self.token_idx,
            "sentence_idx": self.sentence_idx,
            "char_span": [self.char_start, self.char_end],
        }
    
    def to_named_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with named features."""
        result = {
            "token": self.token,
            "lemma": self.lemma,
        }
        for i, name in enumerate(FEATURE_NAMES):
            if i < len(self.vector):
                result[name] = self.vector[i]
        return result
    
    def get_feature(self, name: str) -> Optional[float]:
        """Get feature by name."""
        if name in FEATURE_NAMES:
            idx = FEATURE_NAMES.index(name)
            if idx < len(self.vector):
                return self.vector[idx]
        return None


# ==========================================
# Feature Extractor
# ==========================================

class FeatureExtractor:
    """
    Extracts 20-dimensional feature vectors from text.
    
    Usage:
        extractor = FeatureExtractor()
        features = extractor.extract_text("Oda Nobunaga was born in 1534.")
        
        for feat in features:
            print(f"{feat.token}: {feat.vector}")
    """
    
    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        use_spacy: bool = True,
        dict_dir: Optional[Path] = None,
    ):
        """
        Initialize feature extractor.
        
        Args:
            spacy_model: spaCy model name (default: en_core_web_sm)
            use_spacy: Whether to use spaCy (disable for faster but less accurate)
            dict_dir: Path to dictionary directory
        """
        self.spacy_model_name = spacy_model
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self._nlp = None
        
        # Dictionary provider
        self._dict = DictionaryProvider.get_instance(dict_dir)
        
        # Stats
        self._stats = {
            "tokens_processed": 0,
            "sentences_processed": 0,
            "texts_processed": 0,
        }
    
    @property
    def nlp(self):
        """Lazy load spaCy model."""
        if self._nlp is None and self.use_spacy:
            try:
                self._nlp = spacy.load(self.spacy_model_name)
                print(f"[FeatureExtractor] Loaded spaCy: {self.spacy_model_name}")
            except OSError:
                print(f"[FeatureExtractor] Warning: spaCy model not found. "
                      f"Run: python -m spacy download {self.spacy_model_name}")
                self.use_spacy = False
        return self._nlp
    
    def extract_text(
        self,
        text: str,
        section_idx: int = 0,
        total_sections: int = 1,
    ) -> List[TokenFeature]:
        """
        Extract features from text.
        
        Args:
            text: Input text (can be multi-sentence, multi-paragraph)
            section_idx: Current section index (0-based)
            total_sections: Total number of sections
            
        Returns:
            List of TokenFeature for each token
        """
        if not text or not text.strip():
            return []
        
        self._stats["texts_processed"] += 1
        
        if self.use_spacy and self.nlp:
            return self._extract_spacy(text, section_idx, total_sections)
        else:
            return self._extract_basic(text, section_idx, total_sections)
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs by blank lines or double newlines.
        
        Returns list of non-empty paragraph strings.
        """
        # Split by blank lines (one or more empty lines)
        paragraphs = re.split(r'\n\s*\n', text)
        # Filter empty and strip
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _extract_spacy(
        self,
        text: str,
        section_idx: int,
        total_sections: int,
    ) -> List[TokenFeature]:
        """Extract using spaCy (full features)."""
        results = []
        
        # Split into paragraphs first
        paragraphs = self._split_paragraphs(text)
        if not paragraphs:
            paragraphs = [text.strip()] if text.strip() else []
        
        paragraph_count = len(paragraphs)
        
        # Pre-compute total sentences for normalization
        full_doc = self.nlp(text)
        total_sentences = len(list(full_doc.sents))
        if total_sentences == 0:
            total_sentences = 1
        
        global_sent_idx = 0  # Track sentence index across paragraphs
        
        for para_idx, para_text in enumerate(paragraphs):
            if not para_text.strip():
                continue
            
            doc = self.nlp(para_text)
            sentences = list(doc.sents)
            sentences_in_paragraph = len(sentences)
            
            for sent in sentences:
                tokens = list(sent)
                sent_len = len(tokens)
                
                # Pre-scan: find be-aux positions for passive detection
                be_aux_positions = set()
                for i, tok in enumerate(tokens):
                    if tok.text.lower() in BE_AUXILIARIES:
                        be_aux_positions.add(i)
                
                # Context tracking
                paren_depth = 0
                quote_open = False
                
                for tok_idx, tok in enumerate(tokens):
                    # Update context
                    if tok.text == '(':
                        paren_depth += 1
                    elif tok.text == ')':
                        paren_depth = max(0, paren_depth - 1)
                    if tok.text in ('"', "'", '"', '"'):
                        quote_open = not quote_open
                    
                    # Build vector
                    vector = self._build_vector(
                        token=tok.text,
                        lemma=tok.lemma_,
                        tok_idx=tok_idx,
                        sent_len=sent_len,
                        sent_idx=global_sent_idx,
                        total_sentences=total_sentences,
                        section_idx=section_idx,
                        total_sections=total_sections,
                        paren_depth=paren_depth,
                        quote_open=quote_open,
                        be_aux_positions=be_aux_positions,
                        spacy_token=tok,
                    )
                    
                    feat = TokenFeature(
                        token=tok.text,
                        lemma=tok.lemma_,
                        vector=vector,
                        token_idx=tok_idx,
                        sentence_idx=global_sent_idx,
                        char_start=tok.idx,
                        char_end=tok.idx + len(tok.text),
                        # New paragraph/structure fields
                        paragraph_idx=para_idx,
                        paragraph_count=paragraph_count,
                        sentences_in_paragraph=sentences_in_paragraph,
                        sentence_length=sent_len,
                        total_sentences=total_sentences,
                        total_sections=total_sections,
                    )
                    results.append(feat)
                    self._stats["tokens_processed"] += 1
                
                global_sent_idx += 1
                self._stats["sentences_processed"] += 1
        
        return results
    
    def _extract_basic(
        self,
        text: str,
        section_idx: int,
        total_sections: int,
    ) -> List[TokenFeature]:
        """Extract without spaCy (limited features)."""
        results = []
        
        # Split into paragraphs first
        paragraphs = self._split_paragraphs(text)
        if not paragraphs:
            paragraphs = [text.strip()] if text.strip() else []
        
        paragraph_count = len(paragraphs)
        
        # Count total sentences for normalization
        all_sentences = re.split(r'(?<=[.!?])\s+', text)
        total_sentences = len([s for s in all_sentences if s.strip()])
        if total_sentences == 0:
            total_sentences = 1
        
        global_sent_idx = 0
        
        for para_idx, para_text in enumerate(paragraphs):
            if not para_text.strip():
                continue
            
            sentences = re.split(r'(?<=[.!?])\s+', para_text)
            sentences = [s for s in sentences if s.strip()]
            sentences_in_paragraph = len(sentences)
            
            for sent in sentences:
                tokens = re.findall(r'\S+', sent)
                sent_len = len(tokens)
                
                be_aux_positions = set()
                for i, tok in enumerate(tokens):
                    cleaned = re.sub(r'[^\w]', '', tok).lower()
                    if cleaned in BE_AUXILIARIES:
                        be_aux_positions.add(i)
                
                paren_depth = 0
                quote_open = False
                char_offset = 0
                
                for tok_idx, tok in enumerate(tokens):
                    paren_depth += tok.count('(') - tok.count(')')
                    paren_depth = max(0, paren_depth)
                    if '"' in tok or "'" in tok:
                        quote_open = not quote_open
                    
                    cleaned = re.sub(r'[^\w]', '', tok)
                    lemma = cleaned.lower()
                    
                    vector = self._build_vector(
                        token=tok,
                        lemma=lemma,
                        tok_idx=tok_idx,
                        sent_len=sent_len,
                        sent_idx=global_sent_idx,
                        total_sentences=total_sentences,
                        section_idx=section_idx,
                        total_sections=total_sections,
                        paren_depth=paren_depth,
                        quote_open=quote_open,
                        be_aux_positions=be_aux_positions,
                        spacy_token=None,
                    )
                    
                    feat = TokenFeature(
                        token=tok,
                        lemma=lemma,
                        vector=vector,
                        token_idx=tok_idx,
                        sentence_idx=global_sent_idx,
                        char_start=char_offset,
                        char_end=char_offset + len(tok),
                        # New paragraph/structure fields
                        paragraph_idx=para_idx,
                        paragraph_count=paragraph_count,
                        sentences_in_paragraph=sentences_in_paragraph,
                        sentence_length=sent_len,
                        total_sentences=total_sentences,
                        total_sections=total_sections,
                    )
                    results.append(feat)
                    
                    char_offset += len(tok) + 1
                    self._stats["tokens_processed"] += 1
                
                global_sent_idx += 1
                self._stats["sentences_processed"] += 1
        
        return results
    
    def _build_vector(
        self,
        token: str,
        lemma: str,
        tok_idx: int,
        sent_len: int,
        sent_idx: int,
        total_sentences: int,
        section_idx: int,
        total_sections: int,
        paren_depth: int,
        quote_open: bool,
        be_aux_positions: set,
        spacy_token: Optional["Token"],
    ) -> Tuple[float, ...]:
        """Build 20-dimensional feature vector."""
        
        tok_clean = re.sub(r'[^\w]', '', token)
        tok_clean_lower = tok_clean.lower()
        
        # === Part A: Structural (0-9) ===
        f0 = tok_idx / max(sent_len - 1, 1)
        f1 = sent_idx / max(total_sentences - 1, 1)
        f2 = section_idx / max(total_sections - 1, 1)
        f3 = 1.0 if tok_clean and tok_clean[0].isupper() else 0.0
        f4 = 1.0 if YEAR_PATTERN.match(tok_clean) else 0.0
        f5 = (sent_len - 1 - tok_idx) / max(sent_len - 1, 1)
        f6 = 1.0 if tok_clean_lower in BE_AUXILIARIES else 0.0
        f7 = self._check_passive_participle(tok_clean_lower, tok_idx, be_aux_positions, spacy_token)
        f8 = 1.0 if paren_depth > 0 else 0.0
        f9 = 1.0 if quote_open else 0.0
        
        # === Part B: Psycholinguistic (10-19) ===
        f10 = self._dict.get_score(tok_clean_lower, DICT_CONCRETENESS, lemma)
        f11 = self._dict.get_score(tok_clean_lower, DICT_AOA, lemma)
        f12 = self._dict.get_score(tok_clean_lower, DICT_SENSORIMOTOR, lemma)
        f13 = self._dict.get_score(tok_clean_lower, DICT_VALENCE, lemma)
        
        if spacy_token is not None:
            f14 = 1.0 if spacy_token.pos_ == "PROPN" else 0.0
            is_verb = spacy_token.pos_ == "VERB"
            is_aux = spacy_token.lemma_.lower() in AUX_VERBS
            f15 = 1.0 if (is_verb and not is_aux) else 0.0
            head_dist = abs(spacy_token.head.i - spacy_token.i)
            f18 = head_dist / max(sent_len, 1)
        else:
            f14 = 1.0 if (f3 == 1.0 and tok_idx > 0) else 0.0
            f15 = 0.0
            f18 = NULL_SCORE
        
        f16 = 1.0 if tok_clean_lower.endswith(LATIN_SUFFIXES) else 0.0
        f17 = self._check_numeric(tok_clean, tok_clean_lower, f4)
        f19 = self._check_by_agent(tok_clean_lower, tok_idx, be_aux_positions)
        
        return (f0, f1, f2, f3, f4, f5, f6, f7, f8, f9,
                f10, f11, f12, f13, f14, f15, f16, f17, f18, f19)
    
    def _check_passive_participle(
        self,
        tok_lower: str,
        tok_idx: int,
        be_aux_positions: set,
        spacy_token: Optional["Token"],
    ) -> float:
        """Check if token is passive participle."""
        has_be_before = any(
            i in be_aux_positions
            for i in range(max(0, tok_idx - 2), tok_idx)
        )
        
        if not has_be_before:
            return 0.0
        
        if tok_lower in IRREGULAR_PARTICIPLES:
            return 1.0
        
        if ED_SUFFIX_PATTERN.search(tok_lower):
            return 1.0
        
        if spacy_token is not None and spacy_token.tag_ == "VBN":
            return 1.0
        
        return 0.0
    
    def _check_numeric(self, tok: str, tok_lower: str, is_year: float) -> float:
        """Check if token is numeric (non-year)."""
        if is_year > 0:
            return 0.0
        if DIGIT_PATTERN.search(tok):
            return 1.0
        if tok_lower in NUMBER_WORDS:
            return 1.0
        return 0.0
    
    def _check_by_agent(self, tok_lower: str, tok_idx: int, be_aux_positions: set) -> float:
        """Check if 'by' introduces passive agent."""
        if tok_lower != "by":
            return 0.0
        has_passive_context = any(
            i in be_aux_positions
            for i in range(max(0, tok_idx - 3), tok_idx)
        )
        return 1.0 if has_passive_context else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        return {
            "version": FEATURE_EXTRACTOR_VERSION,
            "feature_dim": FEATURE_DIM,
            "spacy_enabled": self.use_spacy,
            "spacy_model": self.spacy_model_name if self.use_spacy else None,
            **self._stats,
            "dict_stats": self._dict.get_stats(),
        }
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Get list of feature names."""
        return list(FEATURE_NAMES)


# ==========================================
# Batch Processing
# ==========================================

def extract_batch(
    texts: List[str],
    extractor: Optional[FeatureExtractor] = None,
) -> Iterator[Tuple[int, List[TokenFeature]]]:
    """
    Extract features from multiple texts.
    
    Yields:
        (text_idx, features) tuples
    """
    if extractor is None:
        extractor = FeatureExtractor()
    
    total = len(texts)
    for idx, text in enumerate(texts):
        features = extractor.extract_text(text, section_idx=idx, total_sections=total)
        yield idx, features
