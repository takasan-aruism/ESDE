"""
ESDE Engine v5.3.2 - Phase 7B-online: Search Cache

Caches search results to avoid redundant API calls.
"""
import json
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta
from pathlib import Path


class SearchCache:
    """
    Caches search results by token+query.
    
    Cache entry format:
    {
        "key": str,           # hash(token|query)
        "token": str,
        "query": str,
        "results": [...],
        "created_at": str,
        "expires_at": str,
        "hit_count": int
    }
    """
    
    DEFAULT_TTL_HOURS = 24 * 7  # 1 week
    
    def __init__(self, cache_dir: str = "./data/cache", ttl_hours: int = None):
        self.cache_dir = Path(cache_dir)
        self.ttl = timedelta(hours=ttl_hours or self.DEFAULT_TTL_HOURS)
        self._memory_cache: Dict[str, Dict] = {}
        self._ensure_directory()
    
    def _ensure_directory(self):
        """Create cache directory if needed."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _compute_key(self, token: str, query: str) -> str:
        """Compute cache key from token and query."""
        key_str = f"{token.lower()}|{query.lower()}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache entry."""
        # Use first 2 chars as subdirectory for better file distribution
        subdir = key[:2]
        return self.cache_dir / subdir / f"{key}.json"
    
    def get(self, token: str, query: str) -> Optional[Dict[str, Any]]:
        """
        Get cached results for token+query.
        Returns None if not found or expired.
        """
        key = self._compute_key(token, query)
        
        # Check memory cache first
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if self._is_valid(entry):
                entry["hit_count"] = entry.get("hit_count", 0) + 1
                return entry
            else:
                del self._memory_cache[key]
        
        # Check file cache
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    entry = json.load(f)
                
                if self._is_valid(entry):
                    entry["hit_count"] = entry.get("hit_count", 0) + 1
                    self._memory_cache[key] = entry
                    return entry
                else:
                    # Remove expired entry
                    cache_path.unlink()
            except Exception:
                pass
        
        return None
    
    def set(self, token: str, query: str, results: List[Dict[str, Any]]) -> str:
        """
        Cache search results.
        Returns the cache key.
        """
        key = self._compute_key(token, query)
        now = datetime.now(timezone.utc)
        
        entry = {
            "key": key,
            "token": token,
            "query": query,
            "results": results,
            "created_at": now.isoformat(),
            "expires_at": (now + self.ttl).isoformat(),
            "hit_count": 0
        }
        
        # Save to memory
        self._memory_cache[key] = entry
        
        # Save to file
        cache_path = self._get_cache_path(key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[SearchCache] Write error: {e}")
        
        return key
    
    def _is_valid(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid."""
        try:
            expires_at = datetime.fromisoformat(entry["expires_at"].replace('Z', '+00:00'))
            return datetime.now(timezone.utc) < expires_at
        except Exception:
            return False
    
    def invalidate(self, token: str, query: str) -> bool:
        """Invalidate a specific cache entry."""
        key = self._compute_key(token, query)
        
        # Remove from memory
        if key in self._memory_cache:
            del self._memory_cache[key]
        
        # Remove from file
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()
            return True
        
        return False
    
    def clear_expired(self) -> int:
        """Clear all expired entries. Returns count of cleared entries."""
        cleared = 0
        
        # Clear memory cache
        expired_keys = [
            k for k, v in self._memory_cache.items()
            if not self._is_valid(v)
        ]
        for k in expired_keys:
            del self._memory_cache[k]
            cleared += 1
        
        # Clear file cache
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir():
                for cache_file in subdir.glob("*.json"):
                    try:
                        with open(cache_file, 'r') as f:
                            entry = json.load(f)
                        if not self._is_valid(entry):
                            cache_file.unlink()
                            cleared += 1
                    except Exception:
                        pass
        
        return cleared
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        memory_count = len(self._memory_cache)
        file_count = 0
        total_hits = 0
        
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir():
                for cache_file in subdir.glob("*.json"):
                    file_count += 1
                    try:
                        with open(cache_file, 'r') as f:
                            entry = json.load(f)
                        total_hits += entry.get("hit_count", 0)
                    except Exception:
                        pass
        
        return {
            "memory_entries": memory_count,
            "file_entries": file_count,
            "total_hits": total_hits,
            "cache_dir": str(self.cache_dir)
        }
