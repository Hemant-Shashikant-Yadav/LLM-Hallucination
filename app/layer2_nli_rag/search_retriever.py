"""
Layer 2: Async Search Retriever for Evidence Collection.

Provides asynchronous web search (DuckDuckGo) and local vector search
(ChromaDB) to retrieve evidence documents for NLI verification of
Atomic Propositions.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from loguru import logger

from app.config import settings
from app.models.schemas import AtomicProposition, SearchResult


class SearchRetriever:
    """
    Async evidence retriever combining web search and local vector search.

    Supports DuckDuckGo (free, no API key) as the primary backend,
    with optional ChromaDB for cached/domain-specific documents.
    """

    def __init__(
        self,
        backend: str | None = None,
        max_results: int | None = None,
    ) -> None:
        self._backend = backend or settings.search_backend
        self._max_results = max_results or settings.max_search_results
        self._chroma_client = None
        self._chroma_collection = None
        logger.info(
            "SearchRetriever initialized | backend={} | max_results={}",
            self._backend,
            self._max_results,
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def search(
        self,
        query: str,
        max_results: int | None = None,
    ) -> list[SearchResult]:
        """
        Search for evidence related to the given query.

        Args:
            query: Search query string.
            max_results: Override max results.

        Returns:
            List of SearchResult objects ranked by relevance.
        """
        limit = max_results or self._max_results
        results: list[SearchResult] = []

        try:
            if self._backend in ("duckduckgo", "both"):
                web_results = await self._search_duckduckgo(query, limit)
                results.extend(web_results)

            if self._backend in ("tavily", "both"):
                tavily_results = await self._search_tavily(query, limit)
                results.extend(tavily_results)

        except Exception as exc:
            logger.error("Web search failed: {}", exc)

        # Also search local ChromaDB if initialized
        try:
            local_results = await self._search_chromadb(query, limit)
            results.extend(local_results)
        except Exception:
            pass  # ChromaDB is optional

        # Deduplicate and sort by relevance
        results = self._deduplicate(results)
        results.sort(key=lambda r: r.relevance_score, reverse=True)

        logger.info("Retrieved {} search results for query: {:.50}...", len(results), query)
        return results[:limit]

    async def search_for_propositions(
        self,
        propositions: list[AtomicProposition],
    ) -> dict[int, list[SearchResult]]:
        """
        Search for evidence for multiple Atomic Propositions concurrently.

        Args:
            propositions: List of APs to find evidence for.

        Returns:
            Dict mapping proposition ID → list of search results.
        """
        tasks = [
            self.search(prop.text)
            for prop in propositions
        ]

        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        evidence_map: dict[int, list[SearchResult]] = {}
        for prop, results in zip(propositions, results_list):
            if isinstance(results, Exception):
                logger.warning("Search failed for AP {}: {}", prop.id, results)
                evidence_map[prop.id] = []
            else:
                evidence_map[prop.id] = results

        return evidence_map

    # -------------------------------------------------------------------------
    # DuckDuckGo Backend
    # -------------------------------------------------------------------------

    async def _search_duckduckgo(
        self,
        query: str,
        max_results: int,
    ) -> list[SearchResult]:
        """Search using the duckduckgo-search library."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._ddg_search_sync,
            query,
            max_results,
        )

    @staticmethod
    def _ddg_search_sync(query: str, max_results: int) -> list[SearchResult]:
        """Synchronous DuckDuckGo search — called from executor."""
        try:
            from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(
                        SearchResult(
                            title=r.get("title", ""),
                            url=r.get("href", ""),
                            snippet=r.get("body", ""),
                            relevance_score=0.7,  # DuckDuckGo doesn't provide scores
                        )
                    )
            return results

        except ImportError:
            logger.error("duckduckgo-search not installed")
            return []
        except Exception as exc:
            logger.error("DuckDuckGo search error: {}", exc)
            return []

    # -------------------------------------------------------------------------
    # Tavily Backend (Optional)
    # -------------------------------------------------------------------------

    async def _search_tavily(
        self,
        query: str,
        max_results: int,
    ) -> list[SearchResult]:
        """Search using the Tavily API (requires API key)."""
        if not settings.tavily_api_key:
            logger.warning("Tavily API key not configured, skipping Tavily search")
            return []

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": settings.tavily_api_key,
                        "query": query,
                        "max_results": max_results,
                        "search_depth": "basic",
                        "include_answer": False,
                    },
                )
                response.raise_for_status()
                data = response.json()

            results = []
            for r in data.get("results", []):
                results.append(
                    SearchResult(
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        snippet=r.get("content", ""),
                        relevance_score=r.get("score", 0.5),
                    )
                )
            return results

        except Exception as exc:
            logger.error("Tavily search error: {}", exc)
            return []

    # -------------------------------------------------------------------------
    # ChromaDB Local Search
    # -------------------------------------------------------------------------

    async def init_chromadb(self) -> None:
        """Initialize ChromaDB client and collection for local search."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._init_chromadb_sync)

    def _init_chromadb_sync(self) -> None:
        """Synchronous ChromaDB initialization."""
        try:
            import chromadb

            self._chroma_client = chromadb.PersistentClient(
                path=settings.chromadb_path
            )
            self._chroma_collection = self._chroma_client.get_or_create_collection(
                name="evidence_store",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "ChromaDB initialized at {} | collection docs={}",
                settings.chromadb_path,
                self._chroma_collection.count(),
            )
        except ImportError:
            logger.warning("ChromaDB not installed, local search unavailable")
        except Exception as exc:
            logger.warning("ChromaDB init failed: {}", exc)

    async def _search_chromadb(
        self,
        query: str,
        max_results: int,
    ) -> list[SearchResult]:
        """Search the local ChromaDB vector store."""
        if self._chroma_collection is None:
            return []

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._chromadb_query_sync,
            query,
            max_results,
        )

    def _chromadb_query_sync(
        self,
        query: str,
        max_results: int,
    ) -> list[SearchResult]:
        """Synchronous ChromaDB query."""
        if self._chroma_collection is None or self._chroma_collection.count() == 0:
            return []

        try:
            results = self._chroma_collection.query(
                query_texts=[query],
                n_results=min(max_results, self._chroma_collection.count()),
            )

            search_results = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    distance = results["distances"][0][i] if results["distances"] else 0.5
                    relevance = max(0.0, 1.0 - distance)  # Convert distance to similarity
                    metadata = (
                        results["metadatas"][0][i]
                        if results["metadatas"]
                        else {}
                    )
                    search_results.append(
                        SearchResult(
                            title=metadata.get("title", "Local Document"),
                            url=metadata.get("source", ""),
                            snippet=doc,
                            relevance_score=relevance,
                        )
                    )
            return search_results

        except Exception as exc:
            logger.error("ChromaDB query error: {}", exc)
            return []

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def _deduplicate(results: list[SearchResult]) -> list[SearchResult]:
        """Remove duplicate search results based on URL."""
        seen_urls: set[str] = set()
        unique: list[SearchResult] = []
        for r in results:
            key = r.url or r.snippet[:100]
            if key not in seen_urls:
                seen_urls.add(key)
                unique.append(r)
        return unique
