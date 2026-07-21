"""
OncoMOA — Persistent API Cache
SQLite-backed diskcache with 7-day TTL for all API calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
from functools import wraps
from typing import Any, Callable

import diskcache

from config import CACHE_DIR, CACHE_TTL

logger = logging.getLogger(__name__)

# Global cache instance (shared across all tools)
_cache: diskcache.Cache | None = None


def get_cache() -> diskcache.Cache:
    """Return (or initialize) the global diskcache instance."""
    global _cache
    if _cache is None:
        _cache = diskcache.Cache(str(CACHE_DIR), timeout=1)
    return _cache


def make_cache_key(namespace: str, **kwargs: Any) -> str:
    """Generate a deterministic SHA-256 cache key from namespace + query params."""
    payload = json.dumps({"ns": namespace, **kwargs}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def cached_get(key: str) -> Any | None:
    """Retrieve a cached value; returns None on miss or error."""
    try:
        cache = get_cache()
        return cache.get(key)
    except Exception as exc:
        logger.debug("Cache read miss/error for key %s: %s", key[:16], exc)
        return None


def cached_set(key: str, value: Any, ttl: int = CACHE_TTL) -> None:
    """Store a value in the cache with TTL. Silently ignores errors."""
    try:
        cache = get_cache()
        cache.set(key, value, expire=ttl)
    except Exception as exc:
        logger.debug("Cache write error for key %s: %s", key[:16], exc)


def _is_cacheable_api_result(result: Any) -> bool:
    """Return whether an API result contains data safe to reuse from cache."""
    return result is not None and result != [] and result != {}


def cached_api_call(namespace: str, ttl: int = CACHE_TTL):
    """
    Decorator factory for async API functions.

    Usage:
        @cached_api_call("pubmed")
        async def fetch_pubmed(query: str, max_results: int) -> list[dict]:
            ...

    The cache key is derived from the namespace + all call arguments.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = make_cache_key(namespace, args=args, kwargs=kwargs)
            cached = cached_get(key)
            if cached is not None:
                logger.debug("Cache HIT  [%s] key=%s...", namespace, key[:12])
                return cached
            logger.debug("Cache MISS [%s] key=%s...", namespace, key[:12])
            result = await func(*args, **kwargs)
            # Tool clients generally return empty containers after a transport
            # error. Do not persist those ambiguous results for seven days.
            if _is_cacheable_api_result(result):
                cached_set(key, result, ttl=ttl)
            return result
        return wrapper
    return decorator


def clear_cache() -> None:
    """Clear all cached entries (useful for testing)."""
    get_cache().clear()
    logger.info("Cache cleared.")
