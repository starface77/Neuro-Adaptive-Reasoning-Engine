"""Reasoning cache for intermediate results.

Caches:
- LLM reasoning steps (thinking tokens)
- Partial solutions
- Tool execution results
- Planning outputs

This reduces token usage by reusing intermediate results when queries are similar.
"""

import hashlib
import json
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

class ReasoningCache:
    """Cache for intermediate reasoning results."""

    def __init__(self, cache_dir: str = ".nare_cache", ttl: int = 3600):
        """Initialize reasoning cache.

        Args:
            cache_dir: Directory to store cache files
            ttl: Time-to-live in seconds (default 1 hour)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = ttl
        self.memory_cache: Dict[str, Dict[str, Any]] = {}

    def _hash_key(self, query: str, context: Optional[str] = None) -> str:
        """Generate cache key from query and context."""
        content = query
        if context:
            content += f"|{context}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, query: str, context: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get cached result.

        Args:
            query: Query string
            context: Optional context (chat history, repo map)

        Returns:
            Cached result or None if not found/expired
        """
        key = self._hash_key(query, context)

        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['data']
            else:

                del self.memory_cache[key]

        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    entry = json.load(f)
                if time.time() - entry['timestamp'] < self.ttl:

                    self.memory_cache[key] = entry
                    return entry['data']
                else:

                    cache_file.unlink()
            except Exception:
                pass

        return None

    def set(self, query: str, data: Dict[str, Any], context: Optional[str] = None) -> None:
        """Store result in cache.

        Args:
            query: Query string
            data: Data to cache
            context: Optional context
        """
        key = self._hash_key(query, context)
        entry = {
            'timestamp': time.time(),
            'data': data,
        }

        self.memory_cache[key] = entry

        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(entry, f)
        except Exception:
            pass

    def invalidate(self, query: str, context: Optional[str] = None) -> None:
        """Invalidate cached entry.

        Args:
            query: Query string
            context: Optional context
        """
        key = self._hash_key(query, context)

        if key in self.memory_cache:
            del self.memory_cache[key]

        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            cache_file.unlink()

    def clear_expired(self) -> int:
        """Clear all expired entries.

        Returns:
            Number of entries cleared
        """
        cleared = 0
        now = time.time()

        expired_keys = [
            k for k, v in self.memory_cache.items()
            if now - v['timestamp'] >= self.ttl
        ]
        for key in expired_keys:
            del self.memory_cache[key]
            cleared += 1

        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    entry = json.load(f)
                if now - entry['timestamp'] >= self.ttl:
                    cache_file.unlink()
                    cleared += 1
            except Exception:

                cache_file.unlink()
                cleared += 1

        return cleared

    def clear_all(self) -> None:
        """Clear entire cache."""
        self.memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache stats
        """
        disk_entries = len(list(self.cache_dir.glob("*.json")))
        memory_entries = len(self.memory_cache)

        return {
            'memory_entries': memory_entries,
            'disk_entries': disk_entries,
            'ttl': self.ttl,
        }
