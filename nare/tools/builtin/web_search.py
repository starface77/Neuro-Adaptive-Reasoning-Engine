"""Web search tool for NARE agents.

Allows agents to search Google/DuckDuckGo for solutions to errors,
package installation issues, and general programming questions.
"""

import requests
import json
from nare.utils.logger import get_logger
import hashlib
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

log = get_logger("nare.tools.builtin.web_search")

class WebSearchCache:
    """Cache for web search results."""

    def __init__(self, cache_dir: str = ".nare_cache/search", ttl: int = 86400):
        """Initialize search cache.

        Args:
            cache_dir: Directory to store cache files
            ttl: Time-to-live in seconds (default 24 hours)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl

    def _hash_query(self, query: str) -> str:
        """Generate cache key from query."""
        return hashlib.sha256(query.encode()).hexdigest()[:16]

    def get(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results."""
        key = self._hash_query(query)
        cache_file = self.cache_dir / f"{key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if time.time() - data['timestamp'] < self.ttl:
                    return data['results']
                else:
                    cache_file.unlink()
            except Exception:
                pass
        return None

    def set(self, query: str, results: List[Dict[str, Any]]) -> None:
        """Cache search results."""
        key = self._hash_query(query)
        cache_file = self.cache_dir / f"{key}.json"

        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': time.time(),
                    'query': query,
                    'results': results,
                }, f)
        except Exception as e:
            log.warning(f"Failed to cache search results: {e}")

class WebSearch:
    """Web search interface for agents."""

    def __init__(self, provider: str = "duckduckgo", cache_ttl: int = 86400):
        """Initialize web search.

        Args:
            provider: Search provider (duckduckgo, google)
            cache_ttl: Cache TTL in seconds
        """
        self.provider = provider.lower()
        self.cache = WebSearchCache(ttl=cache_ttl)

    def search(
        self,
        query: str,
        max_results: int = 5,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search the web.

        Args:
            query: Search query
            max_results: Maximum number of results
            use_cache: Use cached results if available

        Returns:
            List of search results with keys: title, url, snippet
        """

        if use_cache:
            cached = self.cache.get(query)
            if cached:
                log.info(f"Using cached search results for: {query}")
                return cached[:max_results]

        if self.provider == "duckduckgo":
            results = self._search_duckduckgo(query, max_results)
        elif self.provider == "google":
            results = self._search_google(query, max_results)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        if results:
            self.cache.set(query, results)

        return results

    def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo Instant Answer API."""
        try:

            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': 1,
                'skip_disambig': 1,
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []

            if data.get('Abstract'):
                results.append({
                    'title': data.get('Heading', 'Answer'),
                    'url': data.get('AbstractURL', ''),
                    'snippet': data.get('Abstract', ''),
                })

            for topic in data.get('RelatedTopics', [])[:max_results - len(results)]:
                if isinstance(topic, dict) and 'Text' in topic:
                    results.append({
                        'title': topic.get('Text', '')[:100],
                        'url': topic.get('FirstURL', ''),
                        'snippet': topic.get('Text', ''),
                    })

            if not results:
                results = self._search_duckduckgo_html(query, max_results)

            return results[:max_results]

        except Exception as e:
            log.error(f"DuckDuckGo search failed: {e}")
            return []

    def _search_duckduckgo_html(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Fallback: scrape DuckDuckGo HTML results."""
        try:
            url = "https://html.duckduckgo.com/html/"
            data = {'q': query}
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.post(url, data=data, headers=headers, timeout=10)
            response.raise_for_status()

            html = response.text
            results = []

            import re
            result_pattern = r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]*)</a>'
            snippet_pattern = r'<a[^>]*class="result__snippet"[^>]*>([^<]*)</a>'

            matches = re.findall(result_pattern, html)
            snippets = re.findall(snippet_pattern, html)

            for i, (url, title) in enumerate(matches[:max_results]):
                snippet = snippets[i] if i < len(snippets) else ''
                results.append({
                    'title': title.strip(),
                    'url': url.strip(),
                    'snippet': snippet.strip(),
                })

            return results

        except Exception as e:
            log.error(f"DuckDuckGo HTML scraping failed: {e}")
            return []

    def _search_google(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using Google Custom Search API.

        Note: Requires GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables.
        """
        import os

        api_key = os.getenv('GOOGLE_API_KEY')
        cse_id = os.getenv('GOOGLE_CSE_ID')

        if not api_key or not cse_id:
            log.warning("Google API credentials not found, falling back to DuckDuckGo")
            return self._search_duckduckgo(query, max_results)

        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': api_key,
                'cx': cse_id,
                'q': query,
                'num': max_results,
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get('items', []):
                results.append({
                    'title': item.get('title', ''),
                    'url': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                })

            return results

        except Exception as e:
            log.error(f"Google search failed: {e}")
            return []

    def search_error_solution(self, error_message: str, language: str = "python") -> List[Dict[str, Any]]:
        """Search for solutions to an error message.

        Args:
            error_message: Error message or traceback
            language: Programming language

        Returns:
            List of search results
        """

        error_lines = error_message.strip().split('\n')
        key_error = error_lines[-1] if error_lines else error_message

        query = f"{language} {key_error} solution"

        return self.search(query, max_results=5)

    def search_package_install(self, package_name: str, language: str = "python") -> List[Dict[str, Any]]:
        """Search for package installation instructions.

        Args:
            package_name: Package name
            language: Programming language

        Returns:
            List of search results
        """
        query = f"how to install {package_name} {language}"
        return self.search(query, max_results=3)

_web_search_instance = None

def get_web_search() -> WebSearch:
    """Get singleton WebSearch instance."""
    global _web_search_instance
    if _web_search_instance is None:
        _web_search_instance = WebSearch(provider="duckduckgo")
    return _web_search_instance

def web_search(query: str, max_results: int = 5) -> str:
    """Search the web (tool function for agents).

    Args:
        query: Search query
        max_results: Maximum number of results

    Returns:
        Formatted search results as string
    """
    searcher = get_web_search()
    results = searcher.search(query, max_results=max_results)

    if not results:
        return "No search results found."

    output = []
    for i, result in enumerate(results, 1):
        output.append(f"{i}. {result['title']}")
        output.append(f"   URL: {result['url']}")
        if result['snippet']:
            output.append(f"   {result['snippet']}")
        output.append("")

    return "\n".join(output)
