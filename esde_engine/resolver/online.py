"""
ESDE Engine v5.3.2 - Phase 7B-online: Online Search & Evidence Extraction

Performs web searches and extracts evidence for unknown tokens.
"""
import re
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from ..config import TITLE_LIKE_PHRASES
import random
import time

# =============================================================================
# Source Tiers
# =============================================================================

TIER1_DOMAINS: Set[str] = {
    # Major encyclopedias
    "wikipedia.org", "en.wikipedia.org",
    "britannica.com", "www.britannica.com",
    # Major dictionaries
    "merriam-webster.com", "www.merriam-webster.com",
    "dictionary.com", "www.dictionary.com",
    "oxforddictionaries.com", "lexico.com",
    "cambridge.org", "dictionary.cambridge.org",
    # Other authoritative
    "wordreference.com",
}

TIER2_DOMAINS: Set[str] = {
    # Community/informal
    "wiktionary.org", "en.wiktionary.org",
    "urbandictionary.com", "www.urbandictionary.com",
    # Forums/Q&A
    "stackexchange.com", "english.stackexchange.com",
    "reddit.com", "www.reddit.com",
    # Other
    "yourdictionary.com",
}


def get_source_tier(url: str) -> int:
    """
    Determine source tier from URL.
    Returns: 1 (authoritative), 2 (community), 3 (other)
    """
    url_lower = url.lower()
    
    for domain in TIER1_DOMAINS:
        if domain in url_lower:
            return 1
    
    for domain in TIER2_DOMAINS:
        if domain in url_lower:
            return 2
    
    return 3


# =============================================================================
# Query Generation
# =============================================================================

@dataclass
class SearchQuery:
    """A search query with metadata."""
    query: str
    query_type: str  # definition, slang, typo, phrase, entity
    priority: int = 1  # Higher = more important


def generate_queries(token: str, 
                     context: Dict[str, Any],
                     typo_candidates: List[Dict] = None,
                     route_winner: str = None) -> List[SearchQuery]:
    """
    Generate multiple search queries for a token.
    
    Always generates at least 3 queries (as per spec).
    """
    queries = []
    
    # Q1: Basic definition
    queries.append(SearchQuery(
        query=f'"{token}" meaning',
        query_type="definition",
        priority=1
    ))
    
    # Q2: Dictionary definition
    queries.append(SearchQuery(
        query=f'"{token}" definition',
        query_type="definition",
        priority=1
    ))
    
    # Q3: Slang (for B/C routes)
    if route_winner in ("B", "C", None):
        queries.append(SearchQuery(
            query=f'"{token}" slang meaning',
            query_type="slang",
            priority=2
        ))
    
    # Q4: Typo comparison (if typo candidates exist)
    if typo_candidates:
        best_typo = typo_candidates[0].get("candidate", "")
        if best_typo and best_typo != token:
            queries.append(SearchQuery(
                query=f'"{best_typo}" vs "{token}"',
                query_type="typo",
                priority=2
            ))
            queries.append(SearchQuery(
                query=f'"{token}" misspelling "{best_typo}"',
                query_type="typo",
                priority=3
            ))
    
    # Q5: Title-like phrase search
    phrase = detect_title_phrase(token, context)
    if phrase:
        queries.append(SearchQuery(
            query=f'"{phrase}"',
            query_type="phrase",
            priority=1
        ))
        queries.append(SearchQuery(
            query=f'"{phrase}" song OR movie OR book',
            query_type="phrase",
            priority=2
        ))
    
    # Q6: Entity search (for B route)
    if route_winner == "B":
        queries.append(SearchQuery(
            query=f'"{token}" character OR person OR place',
            query_type="entity",
            priority=2
        ))
    
    # Sort by priority
    queries.sort(key=lambda q: q.priority)
    
    return queries


def detect_title_phrase(token: str, context: Dict[str, Any]) -> Optional[str]:
    """
    Detect if token is part of a title-like phrase.
    Returns the phrase if found, None otherwise.
    """
    tokens_all = context.get("tokens_all", context.get("tokens", []))
    if not tokens_all:
        return None
    
    # Check 2-6 word n-grams
    for n in range(2, min(7, len(tokens_all) + 1)):
        for i in range(len(tokens_all) - n + 1):
            ngram = " ".join(tokens_all[i:i+n]).lower()
            
            # Check against known title patterns
            if ngram in TITLE_LIKE_PHRASES:
                return ngram
            
            # Check if token is in this ngram
            if token.lower() in ngram:
                # Additional heuristics for title detection
                # e.g., "i did it again", "love you"
                common_title_words = {"i", "you", "me", "it", "the", "a"}
                ngram_words = set(ngram.split())
                if len(ngram_words - common_title_words) <= 2:
                    return ngram
    
    return None


# =============================================================================
# Evidence Extraction
# =============================================================================

@dataclass
class EvidenceItem:
    """A piece of evidence extracted from a search result."""
    url: str
    source_tier: int
    title: str
    snippet: str
    published: str = "unknown"
    signals: Dict[str, bool] = field(default_factory=dict)
    claims: List[Dict[str, str]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceCollection:
    """Collection of evidence for a token."""
    token: str
    queries_used: List[str]
    items: List[EvidenceItem] = field(default_factory=list)
    tier1_count: int = 0
    tier2_count: int = 0
    tier3_count: int = 0
    
    def add(self, item: EvidenceItem):
        self.items.append(item)
        if item.source_tier == 1:
            self.tier1_count += 1
        elif item.source_tier == 2:
            self.tier2_count += 1
        else:
            self.tier3_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "token": self.token,
            "queries_used": self.queries_used,
            "items": [item.to_dict() for item in self.items],
            "tier1_count": self.tier1_count,
            "tier2_count": self.tier2_count,
            "tier3_count": self.tier3_count,
            "total_count": len(self.items)
        }


def extract_signals(title: str, snippet: str) -> Dict[str, bool]:
    """
    Extract signals from search result title and snippet.
    """
    text = f"{title} {snippet}".lower()
    
    signals = {
        "is_definition": False,
        "is_work_title": False,
        "is_person": False,
        "is_character": False,
        "is_place": False,
        "mentions_typo": False,
        "mentions_slang": False,
        "mentions_acronym": False,
    }
    
    # Definition signals
    definition_patterns = [
        r"definition", r"meaning", r"means", r"refers to",
        r"is a\b", r"is an\b", r"defined as"
    ]
    for pattern in definition_patterns:
        if re.search(pattern, text):
            signals["is_definition"] = True
            break
    
    # Work title signals
    title_patterns = [
        r"song\b", r"album\b", r"movie\b", r"film\b", r"book\b",
        r"novel\b", r"tv show", r"series\b", r"track\b", r"single\b"
    ]
    for pattern in title_patterns:
        if re.search(pattern, text):
            signals["is_work_title"] = True
            break
    
    # Person signals
    person_patterns = [
        r"born\b", r"died\b", r"actor\b", r"actress\b", r"singer\b",
        r"politician\b", r"author\b", r"writer\b", r"director\b"
    ]
    for pattern in person_patterns:
        if re.search(pattern, text):
            signals["is_person"] = True
            break
    
    # Character signals
    character_patterns = [
        r"character\b", r"fictional\b", r"comic\b", r"superhero\b",
        r"villain\b", r"protagonist\b", r"dc comics", r"marvel"
    ]
    for pattern in character_patterns:
        if re.search(pattern, text):
            signals["is_character"] = True
            break
    
    # Place signals
    place_patterns = [
        r"city\b", r"town\b", r"country\b", r"region\b", r"located\b"
    ]
    for pattern in place_patterns:
        if re.search(pattern, text):
            signals["is_place"] = True
            break
    
    # Typo signals
    typo_patterns = [
        r"misspell", r"typo\b", r"common mistake", r"often confused",
        r"correct spelling", r"incorrect spelling"
    ]
    for pattern in typo_patterns:
        if re.search(pattern, text):
            signals["mentions_typo"] = True
            break
    
    # Slang signals
    slang_patterns = [
        r"slang\b", r"informal\b", r"colloquial", r"internet slang",
        r"urban\b", r"vernacular"
    ]
    for pattern in slang_patterns:
        if re.search(pattern, text):
            signals["mentions_slang"] = True
            break
    
    # Acronym signals
    acronym_patterns = [
        r"acronym\b", r"abbreviation\b", r"stands for", r"short for"
    ]
    for pattern in acronym_patterns:
        if re.search(pattern, text):
            signals["mentions_acronym"] = True
            break
    
    return signals


def extract_claims(token: str, title: str, snippet: str, 
                   signals: Dict[str, bool]) -> List[Dict[str, str]]:
    """
    Extract specific claims from search result.
    """
    claims = []
    text = f"{title} {snippet}"
    
    # Definition claim
    if signals.get("is_definition"):
        # Try to extract the definition
        patterns = [
            rf"{re.escape(token)}\s+(?:is|means|refers to)\s+(.{{20,100}})",
            rf"definition[:\s]+(.{{20,100}})"
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                claims.append({
                    "type": "definition",
                    "text": match.group(1).strip()[:200]
                })
                break
    
    # Entity type claim
    if signals.get("is_character"):
        claims.append({"type": "entity_type", "text": "fictional_character"})
    elif signals.get("is_person"):
        claims.append({"type": "entity_type", "text": "person"})
    elif signals.get("is_place"):
        claims.append({"type": "entity_type", "text": "place"})
    elif signals.get("is_work_title"):
        claims.append({"type": "entity_type", "text": "work_title"})
    
    # Typo/alias claim
    if signals.get("mentions_typo"):
        # Try to extract the correct spelling
        patterns = [
            rf"correct spelling[:\s]+(\w+)",
            rf"should be[:\s]+(\w+)",
            rf"meant[:\s]+(\w+)"
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                claims.append({
                    "type": "alias",
                    "from": token,
                    "to": match.group(1).strip()
                })
                break
    
    return claims


def create_evidence_item(url: str, title: str, snippet: str, 
                         token: str) -> EvidenceItem:
    """
    Create an EvidenceItem from search result data.
    """
    source_tier = get_source_tier(url)
    signals = extract_signals(title, snippet)
    claims = extract_claims(token, title, snippet, signals)
    
    return EvidenceItem(
        url=url,
        source_tier=source_tier,
        title=title[:200],
        snippet=snippet[:500],
        published="unknown",
        signals=signals,
        claims=claims
    )


# =============================================================================
# Search Interface (Abstract - needs concrete implementation)
# =============================================================================

class SearchProvider:
    """
    Abstract interface for search providers.
    
    Concrete implementations can use:
    - Web search APIs (Google, Bing, DuckDuckGo)
    - Local search indices
    - Mock data for testing
    """
    
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Execute a search query.
        
        Returns list of results with at least:
        - url: str
        - title: str
        - snippet: str
        """
        raise NotImplementedError


class MockSearchProvider(SearchProvider):
    """
    Mock search provider for testing.
    Returns predefined results based on patterns.
    """
    
    def __init__(self):
        self.mock_results = {
            "ops": [
                {
                    "url": "https://en.wikipedia.org/wiki/Oops!..._I_Did_It_Again",
                    "title": "Oops!... I Did It Again - Wikipedia",
                    "snippet": "\"Oops!... I Did It Again\" is a song by American singer Britney Spears. 'Ops' is a common misspelling of 'oops'."
                },
                {
                    "url": "https://www.merriam-webster.com/dictionary/oops",
                    "title": "Oops Definition - Merriam-Webster",
                    "snippet": "Oops: used typically to express mild apology, surprise, or dismay. Often misspelled as 'ops'."
                }
            ],
            "superman": [
                {
                    "url": "https://en.wikipedia.org/wiki/Superman",
                    "title": "Superman - Wikipedia",
                    "snippet": "Superman is a fictional superhero appearing in American comic books published by DC Comics."
                },
                {
                    "url": "https://www.britannica.com/topic/Superman-fictional-character",
                    "title": "Superman | Character, Movies, Logo, & Comics | Britannica",
                    "snippet": "Superman, American comic book character created for DC Comics by writer Jerry Siegel and artist Joe Shuster."
                },
                {
                    "url": "https://en.wikipedia.org/wiki/%C3%9Cbermensch",
                    "title": "Übermensch - Wikipedia",
                    "snippet": "The Übermensch (German for 'Superman' or 'Overman') is a concept in the philosophy of Friedrich Nietzsche."
                }
            ]
        }
    
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Return mock results based on query."""
        query_lower = query.lower()
        
        # Find matching mock results
        for key, results in self.mock_results.items():
            if key in query_lower:
                return results[:max_results]
        
        # Default: return empty
        return []


# =============================================================================
# Main Search Function
# =============================================================================

def collect_evidence(token: str,
                     context: Dict[str, Any],
                     search_provider: SearchProvider,
                     typo_candidates: List[Dict] = None,
                     route_winner: str = None,
                     min_sources: int = 3,
                     max_sources: int = 8,
                     cache = None) -> EvidenceCollection:
    """
    Collect evidence for a token using online search.
    
    Args:
        token: The token to search for
        context: Context from queue record
        search_provider: Search provider to use
        typo_candidates: List of typo candidates (optional)
        route_winner: Winner route from 7A+ (optional)
        min_sources: Minimum number of sources to collect
        max_sources: Maximum number of sources to collect
        cache: SearchCache instance (optional)
    
    Returns:
        EvidenceCollection with all gathered evidence
    """
    # Generate queries
    queries = generate_queries(token, context, typo_candidates, route_winner)
    
    collection = EvidenceCollection(
        token=token,
        queries_used=[q.query for q in queries]
    )
    
    seen_urls: Set[str] = set()
    
    for sq in queries:
        # Check cache first
        if cache:
            cached = cache.get(token, sq.query)
            if cached:
                results = cached.get("results", [])
            else:
                results = search_provider.search(sq.query, max_results=5)
                cache.set(token, sq.query, results)
        else:
            results = search_provider.search(sq.query, max_results=5)
        
        # Process results
        for result in results:
            url = result.get("url", "")
            
            # Skip duplicates
            if url in seen_urls:
                continue
            seen_urls.add(url)
            
            # Skip Tier-3 if we have enough Tier-1/2
            tier = get_source_tier(url)
            if tier == 3 and (collection.tier1_count + collection.tier2_count) >= min_sources:
                continue
            
            # Create evidence item
            item = create_evidence_item(
                url=url,
                title=result.get("title", ""),
                snippet=result.get("snippet", ""),
                token=token
            )
            collection.add(item)
            
            # Stop if we have enough
            if len(collection.items) >= max_sources:
                break
        
        # Stop if we have enough
        if len(collection.items) >= max_sources:
            break
    
    return collection


# =============================================================================
# Multi-Source Search Provider (online_v4 based)
# 2026/01/10 - SearXNG から移行
# =============================================================================

import requests

# Try importing ddgs/duckduckgo_search
try:
    from ddgs import DDGS
    DDG_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        DDG_AVAILABLE = True
    except ImportError:
        DDG_AVAILABLE = False


class FreeDictionaryAPI:
    """Free Dictionary API - Wiktionary-based"""
    
    name = "free_dictionary"
    base_url = "https://api.dictionaryapi.dev/api/v2/entries/en"
    timeout = 10
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        headers = {
            "User-Agent": "ESDE-Engine/5.3.5 (Academic Research)",
            "Accept": "application/json",
        }
        try:
            url = f"{self.base_url}/{query.lower().strip()}"
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            results = []
            definitions = []
            
            for entry in data:
                for meaning in entry.get("meanings", []):
                    for defn in meaning.get("definitions", []):
                        if defn.get("definition"):
                            definitions.append(defn["definition"])
            
            if definitions:
                results.append({
                    "url": f"https://en.wiktionary.org/wiki/{query}",
                    "title": f"{query} - Wiktionary",
                    "snippet": " | ".join(definitions[:3])
                })
            
            return results
            
        except Exception:
            return []


class WikipediaAPI:
    """Wikipedia API - Summary extraction"""
    
    name = "wikipedia"
    base_url = "https://en.wikipedia.org/api/rest_v1/page/summary"
    timeout = 10
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        headers = {
            "User-Agent": "ESDE-Engine/5.3.5 (Academic Research)",
            "Accept": "application/json",
        }
        try:
            title = query.strip().replace(" ", "_")
            url = f"{self.base_url}/{title}"
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            extract = data.get("extract", "")
            if extract:
                return [{
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{title}"),
                    "title": data.get("title", query),
                    "snippet": extract[:500]
                }]
            return []
            
        except Exception:
            return []


class DatamuseAPI:
    """Datamuse API - WordNet-based"""
    
    name = "datamuse"
    base_url = "https://api.datamuse.com/words"
    timeout = 10
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        headers = {
            "User-Agent": "ESDE-Engine/5.3.5 (Academic Research)",
        }
        try:
            params = {
                "sp": query.lower().strip(),
                "md": "d,p",
                "max": 5
            }
            response = requests.get(self.base_url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            for word_data in data:
                if word_data.get("word", "").lower() == query.lower():
                    defs = word_data.get("defs", [])
                    if defs:
                        definitions = []
                        for d in defs:
                            if "\t" in d:
                                _, defn = d.split("\t", 1)
                                definitions.append(defn)
                            else:
                                definitions.append(d)
                        
                        return [{
                            "url": f"https://www.onelook.com/?w={query}",
                            "title": f"{query} - Datamuse",
                            "snippet": " | ".join(definitions[:3])
                        }]
            return []
            
        except Exception:
            return []


class UrbanDictionaryAPI:
    """Urban Dictionary - Slang and neologisms"""
    
    name = "urban_dictionary"
    base_url = "https://api.urbandictionary.com/v0/define"
    timeout = 10
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        headers = {
            "User-Agent": "ESDE-Engine/5.3.5 (Academic Research)",
        }
        try:
            params = {"term": query.strip()}
            response = requests.get(self.base_url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            if data.get("list"):
                entries = sorted(data["list"], key=lambda x: x.get("thumbs_up", 0), reverse=True)[:2]
                definitions = []
                for entry in entries:
                    defn = entry.get("definition", "").replace("[", "").replace("]", "")
                    if defn:
                        definitions.append(defn[:200])
                
                if definitions:
                    return [{
                        "url": f"https://www.urbandictionary.com/define.php?term={query}",
                        "title": f"{query} - Urban Dictionary",
                        "snippet": " | ".join(definitions)
                    }]
            return []
            
        except Exception:
            return []


class DuckDuckGoAPI:
    """DuckDuckGo Search - Final fallback"""
    
    name = "duckduckgo"
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        if not DDG_AVAILABLE:
            return []
        
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                with DDGS() as ddgs:
                    search_query = f"{query} definition meaning"
                    results = list(ddgs.text(
                        keywords=search_query,
                        region='wt-wt',
                        safesearch='off',
                        max_results=3
                    ))
                    
                    output = []
                    for r in results:
                        output.append({
                            "url": r.get("href", ""),
                            "title": r.get("title", ""),
                            "snippet": r.get("body", "")
                        })
                    return output
        except Exception:
            return []


class MultiSourceProvider(SearchProvider):
    """
    Multi-source search provider using multiple free APIs.
    Replaces SearXNGProvider with direct API access.
    
    2026/01/10 - online_v4.py based
    """
    
    def __init__(self, request_interval: float = 1.0):
        self.request_interval = request_interval
        self.last_request_time = 0
        
        # Providers in priority order
        self.providers = [
            FreeDictionaryAPI(),
            WikipediaAPI(),
            DatamuseAPI(),
            UrbanDictionaryAPI(),
            DuckDuckGoAPI(),
        ]
        
        # Stats
        self.stats = {p.name: {"success": 0, "fail": 0} for p in self.providers}
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)
        self.last_request_time = time.time()
    
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search using multiple sources with fallback.
        
        Returns results in standard format:
        [{"url": str, "title": str, "snippet": str}, ...]
        """
        # Extract token from query (remove "meaning", "definition" etc)
        token = query.replace('"', '').split()[0] if query else ""
        
        all_results = []
        seen_urls = set()
        
        for provider in self.providers:
            if len(all_results) >= max_results:
                break
            
            self._rate_limit()
            
            try:
                results = provider.search(token)
                
                if results:
                    self.stats[provider.name]["success"] += 1
                    
                    for r in results:
                        url = r.get("url", "")
                        if url not in seen_urls:
                            seen_urls.add(url)
                            all_results.append(r)
                    
                    # Found results, can return early for efficiency
                    if len(all_results) >= 1:
                        return all_results[:max_results]
                else:
                    self.stats[provider.name]["fail"] += 1
                    
            except Exception as e:
                self.stats[provider.name]["fail"] += 1
        
        return all_results[:max_results]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics."""
        return self.stats


# Backward compatibility alias
SearXNGProvider = MultiSourceProvider
