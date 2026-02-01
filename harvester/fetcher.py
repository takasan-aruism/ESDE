"""
ESDE Harvester: Fetcher
========================

HTTP/Wikipedia API access layer.
Returns raw response data without interpretation.

Substrate alignment:
  - Returns machine-observable metadata (status code, content length, etc.)
  - Does NOT classify or interpret content
"""

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


# ==========================================
# Constants
# ==========================================

WIKIPEDIA_API_BASE = "https://{lang}.wikipedia.org/w/api.php"
USER_AGENT = "ESDE/1.0 (research project; harvester)"
FETCH_TIMEOUT = 30


# ==========================================
# FetchResult
# ==========================================

@dataclass
class FetchResult:
    """
    Raw fetch result. No interpretation, just facts.
    
    Substrate-compatible: all fields are machine-observable.
    """
    # Source
    url: str
    title: str
    
    # Response
    success: bool
    status_code: int = 200
    
    # Content (raw)
    raw_json: Optional[Dict[str, Any]] = None  # Full API response
    extract_text: Optional[str] = None          # Extracted plaintext
    
    # Machine-observable metadata
    content_length: int = 0      # len(extract_text)
    section_count: int = 0       # Number of == sections ==
    
    # Timing
    fetched_at: str = ""
    
    # Error
    error_message: Optional[str] = None


# ==========================================
# Wikipedia Fetcher
# ==========================================

def fetch_wikipedia(
    title: str,
    lang: str = "en",
) -> FetchResult:
    """
    Fetch article from Wikipedia API.
    
    Returns FetchResult with raw data and machine-observable metadata.
    Does NOT interpret or classify content.
    
    Args:
        title: Wikipedia article title
        lang: Language code
        
    Returns:
        FetchResult with raw data
    """
    base_url = WIKIPEDIA_API_BASE.format(lang=lang)
    
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": "true",
        "format": "json",
    }
    
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    fetched_at = datetime.now(timezone.utc).isoformat()
    
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT}
        )
        
        with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT) as response:
            raw_bytes = response.read()
            raw_json = json.loads(raw_bytes.decode("utf-8"))
            status_code = response.status
        
        # Extract text from API response
        pages = raw_json.get("query", {}).get("pages", {})
        extract_text = None
        
        for page_id, page_data in pages.items():
            if page_id == "-1":
                return FetchResult(
                    url=url,
                    title=title,
                    success=False,
                    status_code=status_code,
                    fetched_at=fetched_at,
                    error_message=f"Article not found: {title}",
                )
            extract_text = page_data.get("extract", "")
            break
        
        if not extract_text:
            return FetchResult(
                url=url,
                title=title,
                success=False,
                status_code=status_code,
                fetched_at=fetched_at,
                error_message="Empty extract",
            )
        
        # Count sections (machine-observable)
        section_count = extract_text.count("\n== ")
        # Also count subsections
        section_count += extract_text.count("\n=== ")
        
        return FetchResult(
            url=url,
            title=title,
            success=True,
            status_code=status_code,
            raw_json=raw_json,
            extract_text=extract_text,
            content_length=len(extract_text),
            section_count=section_count,
            fetched_at=fetched_at,
        )
        
    except Exception as e:
        return FetchResult(
            url=url,
            title=title,
            success=False,
            fetched_at=fetched_at,
            error_message=str(e),
        )
