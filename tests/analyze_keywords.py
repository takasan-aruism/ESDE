#!/usr/bin/env python3
"""
ESDE Phase 9: Keyword-Centric Structure Analysis
=================================================

Analyzes how important keywords (proper nouns) are structurally
treated within an article.

For each keyword, extracts:
  - Position distribution (where in sentence/section)
  - Sentence length distribution (short fact vs long explanation)
  - Mode rates (parentheses, quotes, passive)
  - Neighboring patterns (what words surround it)
  - Section distribution (which topics mention it)

Usage:
    python tests/analyze_keywords.py
    python tests/analyze_keywords.py --article "Tokugawa Ieyasu"
    python tests/analyze_keywords.py --top-k 20

Spec: Phase 9 Keyword Analysis v1.0
"""

import sys
import time
import json
import re
import argparse
import urllib.request
import urllib.parse
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from statistics.features import (
    FeatureExtractor,
    FEATURE_NAMES,
    NULL_SCORE,
)


# ==========================================
# Data Structures
# ==========================================

@dataclass
class TokenContext:
    """Extended token information for keyword analysis."""
    token: str
    lemma: str
    token_idx: int          # Position in sentence
    sentence_idx: int       # Sentence index in section
    section_idx: int        # Section index in article
    section_name: str       # Section title
    sentence_length: int    # Number of tokens in sentence
    pos_in_sentence: float  # Normalized position [0,1]
    inside_paren: bool
    inside_quote: bool
    is_passive_sentence: bool  # Any passive participle in sentence
    left_context: List[str]    # Tokens to the left
    right_context: List[str]   # Tokens to the right


@dataclass
class KeywordProfile:
    """Structural profile for a keyword."""
    keyword: str
    total_occurrences: int = 0
    
    # Position distribution
    pos_in_sentence_values: List[float] = field(default_factory=list)
    sentence_lengths: List[int] = field(default_factory=list)
    
    # Section distribution
    section_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Mode rates
    paren_count: int = 0
    quote_count: int = 0
    passive_sentence_count: int = 0
    
    # Neighboring patterns
    left_bigrams: Counter = field(default_factory=Counter)
    right_bigrams: Counter = field(default_factory=Counter)
    
    def add_occurrence(self, ctx: TokenContext):
        """Add an occurrence to the profile."""
        self.total_occurrences += 1
        self.pos_in_sentence_values.append(ctx.pos_in_sentence)
        self.sentence_lengths.append(ctx.sentence_length)
        self.section_counts[ctx.section_name] += 1
        
        if ctx.inside_paren:
            self.paren_count += 1
        if ctx.inside_quote:
            self.quote_count += 1
        if ctx.is_passive_sentence:
            self.passive_sentence_count += 1
        
        # Neighboring patterns
        if ctx.left_context:
            left_bigram = " ".join(ctx.left_context[-2:]) if len(ctx.left_context) >= 2 else ctx.left_context[-1]
            self.left_bigrams[left_bigram] += 1
        if ctx.right_context:
            right_bigram = " ".join(ctx.right_context[:2]) if len(ctx.right_context) >= 2 else ctx.right_context[0]
            self.right_bigrams[right_bigram] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics."""
        n = self.total_occurrences
        if n == 0:
            return {}
        
        pos_values = self.pos_in_sentence_values
        sent_lengths = self.sentence_lengths
        
        # Position stats
        pos_mean = sum(pos_values) / n
        pos_variance = sum((x - pos_mean) ** 2 for x in pos_values) / n if n > 1 else 0
        
        # Sentence length stats
        sent_mean = sum(sent_lengths) / n
        
        # Mode rates
        paren_rate = self.paren_count / n
        quote_rate = self.quote_count / n
        passive_rate = self.passive_sentence_count / n
        
        # Section distribution (top 5)
        top_sections = sorted(self.section_counts.items(), key=lambda x: -x[1])[:5]
        
        # Neighboring patterns (top 5)
        top_left = self.left_bigrams.most_common(5)
        top_right = self.right_bigrams.most_common(5)
        
        return {
            "keyword": self.keyword,
            "occurrences": n,
            "position": {
                "mean": round(pos_mean, 3),
                "variance": round(pos_variance, 4),
                "interpretation": "subject/topic" if pos_mean < 0.3 else "mid-sentence" if pos_mean < 0.7 else "end/modifier",
            },
            "sentence_length": {
                "mean": round(sent_mean, 1),
                "interpretation": "short facts" if sent_mean < 15 else "medium" if sent_mean < 25 else "long explanations",
            },
            "modes": {
                "paren_rate": round(paren_rate, 3),
                "quote_rate": round(quote_rate, 3),
                "passive_rate": round(passive_rate, 3),
            },
            "sections": top_sections,
            "left_patterns": top_left,
            "right_patterns": top_right,
        }


# ==========================================
# Wikipedia Fetcher (reuse from stage2)
# ==========================================

def fetch_wikipedia_article(title: str, lang: str = "en") -> Optional[str]:
    """Fetch article text from Wikipedia API."""
    base_url = f"https://{lang}.wikipedia.org/w/api.php"
    
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": "true",
        "format": "json",
    }
    
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    
    try:
        print(f"[Wikipedia] Fetching: {title}")
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "ESDE/1.0 (research project)"}
        )
        
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
        
        pages = data.get("query", {}).get("pages", {})
        
        for page_id, page_data in pages.items():
            if page_id == "-1":
                return None
            return page_data.get("extract", "")
        
        return None
    except Exception as e:
        print(f"[Wikipedia] Error: {e}")
        return None


def split_into_sections(text: str) -> List[Dict[str, Any]]:
    """Split Wikipedia text into sections."""
    heading_pattern = re.compile(r'^(=+)\s*(.+?)\s*\1\s*$', re.MULTILINE)
    
    sections = []
    matches = list(heading_pattern.finditer(text))
    
    # Lead section
    if matches:
        lead_end = matches[0].start()
        lead_content = text[:lead_end].strip()
    else:
        lead_content = text.strip()
    
    if lead_content:
        sections.append({"title": "Lead", "level": 0, "content": lead_content})
    
    skip_titles = {"references", "see also", "external links", "notes", 
                   "further reading", "bibliography", "sources"}
    
    for i, match in enumerate(matches):
        level = len(match.group(1)) - 1
        title = match.group(2).strip()
        
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[content_start:content_end].strip()
        
        # Clean markup
        content = re.sub(r'^(=+)\s*(.+?)\s*\1\s*$', r'\2.', content, flags=re.MULTILINE)
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        if content and title.lower() not in skip_titles:
            sections.append({"title": title, "level": level, "content": content})
    
    return sections


# ==========================================
# Extended Feature Extraction
# ==========================================

def extract_with_context(
    sections: List[Dict[str, Any]],
    context_window: int = 3,
) -> Tuple[List[TokenContext], Dict[str, Any]]:
    """
    Extract tokens with extended context information.
    
    Args:
        sections: List of section dicts
        context_window: Number of tokens to capture on each side
        
    Returns:
        (list of TokenContext, metadata dict)
    """
    extractor = FeatureExtractor(use_spacy=True)
    
    all_contexts = []
    total_sections = len(sections)
    
    for section_idx, section in enumerate(sections):
        section_name = section['title']
        content = section['content']
        
        # Extract features
        features = extractor.extract_text(
            content,
            section_idx=section_idx,
            total_sections=total_sections,
        )
        
        if not features:
            continue
        
        # Group by sentence
        sentences: Dict[int, List] = defaultdict(list)
        for feat in features:
            sentences[feat.sentence_idx].append(feat)
        
        # Process each sentence
        for sent_idx, sent_tokens in sentences.items():
            sent_len = len(sent_tokens)
            
            # Check if sentence has passive
            has_passive = any(t.vector[7] == 1.0 for t in sent_tokens)  # is_passive_participle
            
            for i, feat in enumerate(sent_tokens):
                # Get context
                left_ctx = [sent_tokens[j].token for j in range(max(0, i - context_window), i)]
                right_ctx = [sent_tokens[j].token for j in range(i + 1, min(sent_len, i + 1 + context_window))]
                
                ctx = TokenContext(
                    token=feat.token,
                    lemma=feat.lemma,
                    token_idx=i,
                    sentence_idx=sent_idx,
                    section_idx=section_idx,
                    section_name=section_name,
                    sentence_length=sent_len,
                    pos_in_sentence=feat.vector[0],  # pos_in_sentence_norm
                    inside_paren=feat.vector[8] == 1.0,  # inside_parentheses
                    inside_quote=feat.vector[9] == 1.0,  # is_in_quote
                    is_passive_sentence=has_passive,
                    left_context=left_ctx,
                    right_context=right_ctx,
                )
                all_contexts.append(ctx)
    
    metadata = {
        "total_tokens": len(all_contexts),
        "total_sections": total_sections,
        "total_sentences": extractor.get_stats()['sentences_processed'],
    }
    
    return all_contexts, metadata


# ==========================================
# Keyword Analysis
# ==========================================

def select_keywords(
    contexts: List[TokenContext],
    top_k: int = 15,
    min_count: int = 5,
) -> List[str]:
    """
    Select important keywords based on proper noun frequency.
    
    Args:
        contexts: List of TokenContext
        top_k: Maximum number of keywords
        min_count: Minimum occurrences
        
    Returns:
        List of keyword strings
    """
    # Count proper nouns (is_capitalized and not sentence-initial)
    propn_counts = Counter()
    
    for ctx in contexts:
        # Skip punctuation and numbers
        if not ctx.token.isalpha():
            continue
        
        # Check if capitalized (likely proper noun)
        if ctx.token[0].isupper():
            # Exclude sentence-initial (token_idx == 0)
            if ctx.token_idx > 0:
                propn_counts[ctx.token] += 1
            else:
                # Still count if it appears mid-sentence elsewhere
                # We'll filter by frequency anyway
                propn_counts[ctx.token] += 0.5  # Partial credit
    
    # Filter and sort
    candidates = [
        (word, int(count))
        for word, count in propn_counts.most_common(top_k * 2)
        if count >= min_count
    ]
    
    return [word for word, count in candidates[:top_k]]


def analyze_keywords(
    contexts: List[TokenContext],
    keywords: List[str],
) -> Dict[str, KeywordProfile]:
    """
    Build structural profiles for each keyword.
    
    Args:
        contexts: List of TokenContext
        keywords: List of keywords to analyze
        
    Returns:
        Dict mapping keyword to KeywordProfile
    """
    profiles = {kw: KeywordProfile(keyword=kw) for kw in keywords}
    
    for ctx in contexts:
        if ctx.token in profiles:
            profiles[ctx.token].add_occurrence(ctx)
    
    return profiles


# ==========================================
# Output Formatting
# ==========================================

def print_keyword_report(profiles: Dict[str, KeywordProfile]):
    """Print formatted keyword analysis report."""
    
    print("\n" + "=" * 70)
    print("KEYWORD STRUCTURAL ANALYSIS")
    print("=" * 70)
    
    # Sort by occurrence count
    sorted_profiles = sorted(
        profiles.values(),
        key=lambda p: -p.total_occurrences
    )
    
    for profile in sorted_profiles:
        stats = profile.get_stats()
        if not stats:
            continue
        
        print(f"\n{'─' * 70}")
        print(f"  [{stats['keyword']}]  ({stats['occurrences']} occurrences)")
        print(f"{'─' * 70}")
        
        # Position
        pos = stats['position']
        print(f"\n  Position in sentence:")
        print(f"    mean={pos['mean']:.3f}, variance={pos['variance']:.4f}")
        print(f"    → {pos['interpretation']}")
        
        # Sentence length
        sent = stats['sentence_length']
        print(f"\n  Sentence length (when this word appears):")
        print(f"    mean={sent['mean']:.1f} tokens")
        print(f"    → {sent['interpretation']}")
        
        # Modes
        modes = stats['modes']
        print(f"\n  Structural modes:")
        print(f"    parentheses: {modes['paren_rate']:.1%}")
        print(f"    quotes:      {modes['quote_rate']:.1%}")
        print(f"    passive:     {modes['passive_rate']:.1%}")
        
        # Sections
        print(f"\n  Top sections:")
        for section, count in stats['sections']:
            pct = count / stats['occurrences'] * 100
            print(f"    {section[:40]:40} {count:3} ({pct:.0f}%)")
        
        # Patterns
        print(f"\n  Left patterns (what precedes):")
        for pattern, count in stats['left_patterns']:
            print(f"    '{pattern}' → [{stats['keyword']}]  ({count})")
        
        print(f"\n  Right patterns (what follows):")
        for pattern, count in stats['right_patterns']:
            print(f"    [{stats['keyword']}] → '{pattern}'  ({count})")


def print_comparison_table(profiles: Dict[str, KeywordProfile]):
    """Print comparison table of all keywords."""
    
    print("\n" + "=" * 70)
    print("KEYWORD COMPARISON TABLE")
    print("=" * 70)
    
    print(f"\n  {'Keyword':<15} {'Count':>6} {'Pos':>6} {'SentLen':>8} {'Paren':>6} {'Quote':>6} {'Pass':>6}")
    print(f"  {'-'*15} {'-'*6} {'-'*6} {'-'*8} {'-'*6} {'-'*6} {'-'*6}")
    
    sorted_profiles = sorted(
        profiles.values(),
        key=lambda p: -p.total_occurrences
    )
    
    for profile in sorted_profiles:
        stats = profile.get_stats()
        if not stats:
            continue
        
        kw = stats['keyword'][:15]
        count = stats['occurrences']
        pos = stats['position']['mean']
        sent_len = stats['sentence_length']['mean']
        paren = stats['modes']['paren_rate']
        quote = stats['modes']['quote_rate']
        passive = stats['modes']['passive_rate']
        
        print(f"  {kw:<15} {count:>6} {pos:>6.2f} {sent_len:>8.1f} {paren:>6.1%} {quote:>6.1%} {passive:>6.1%}")


# ==========================================
# Main
# ==========================================

def run_keyword_analysis(
    article_title: str = "Oda Nobunaga",
    top_k: int = 15,
    save_output: bool = False,
):
    """Run keyword-centric structure analysis."""
    
    print("\n" + "#" * 70)
    print("# ESDE Phase 9: Keyword-Centric Structure Analysis")
    print("#" * 70)
    
    # ==========================================
    # 1. Fetch Article
    # ==========================================
    print("\n" + "=" * 60)
    print("[Step 1] Fetch Wikipedia Article")
    print("=" * 60)
    
    text = fetch_wikipedia_article(article_title)
    if not text:
        print("Failed to fetch article.")
        return 1
    
    print(f"  Article: {article_title}")
    print(f"  Characters: {len(text):,}")
    
    # ==========================================
    # 2. Parse Sections
    # ==========================================
    print("\n" + "=" * 60)
    print("[Step 2] Parse Sections")
    print("=" * 60)
    
    sections = split_into_sections(text)
    print(f"  Sections: {len(sections)}")
    
    # ==========================================
    # 3. Extract with Context
    # ==========================================
    print("\n" + "=" * 60)
    print("[Step 3] Extract Tokens with Context")
    print("=" * 60)
    
    start_time = time.time()
    contexts, metadata = extract_with_context(sections)
    elapsed = time.time() - start_time
    
    print(f"  Tokens: {metadata['total_tokens']:,}")
    print(f"  Sentences: {metadata['total_sentences']}")
    print(f"  Time: {elapsed:.2f}s")
    
    # ==========================================
    # 4. Select Keywords
    # ==========================================
    print("\n" + "=" * 60)
    print("[Step 4] Select Important Keywords")
    print("=" * 60)
    
    keywords = select_keywords(contexts, top_k=top_k)
    print(f"  Selected {len(keywords)} keywords:")
    for i, kw in enumerate(keywords):
        print(f"    {i+1:2}. {kw}")
    
    # ==========================================
    # 5. Analyze Keywords
    # ==========================================
    print("\n" + "=" * 60)
    print("[Step 5] Build Keyword Profiles")
    print("=" * 60)
    
    profiles = analyze_keywords(contexts, keywords)
    print(f"  Analyzed {len(profiles)} keywords")
    
    # ==========================================
    # 6. Report
    # ==========================================
    print_comparison_table(profiles)
    print_keyword_report(profiles)
    
    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "=" * 70)
    print("[Summary]")
    print("=" * 70)
    
    print(f"\n  Article: {article_title}")
    print(f"  Total tokens: {metadata['total_tokens']:,}")
    print(f"  Keywords analyzed: {len(keywords)}")
    
    # Interesting findings
    print(f"\n  Structural observations:")
    
    # Find most subject-like (low pos_in_sentence)
    subject_like = min(profiles.values(), key=lambda p: p.get_stats().get('position', {}).get('mean', 1) if p.get_stats() else 1)
    if subject_like.get_stats():
        print(f"    Most subject-like: {subject_like.keyword} (pos={subject_like.get_stats()['position']['mean']:.2f})")
    
    # Find most parenthetical
    most_paren = max(profiles.values(), key=lambda p: p.get_stats().get('modes', {}).get('paren_rate', 0) if p.get_stats() else 0)
    if most_paren.get_stats():
        print(f"    Most in parentheses: {most_paren.keyword} ({most_paren.get_stats()['modes']['paren_rate']:.1%})")
    
    # Find most passive
    most_passive = max(profiles.values(), key=lambda p: p.get_stats().get('modes', {}).get('passive_rate', 0) if p.get_stats() else 0)
    if most_passive.get_stats():
        print(f"    Most in passive sentences: {most_passive.keyword} ({most_passive.get_stats()['modes']['passive_rate']:.1%})")
    
    print("\n" + "#" * 70)
    print("# Keyword Analysis Complete!")
    print("#" * 70)
    
    return 0


def main():
    parser = argparse.ArgumentParser(description="Keyword Structure Analysis")
    parser.add_argument("--article", default="Oda Nobunaga", help="Wikipedia article title")
    parser.add_argument("--top-k", type=int, default=15, help="Number of keywords to analyze")
    parser.add_argument("--save-output", action="store_true", help="Save output files")
    
    args = parser.parse_args()
    
    return run_keyword_analysis(
        article_title=args.article,
        top_k=args.top_k,
        save_output=args.save_output,
    )


if __name__ == "__main__":
    sys.exit(main())
