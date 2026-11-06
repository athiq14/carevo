"""
Modular Web Research Engine for live market and career intelligence.
Supports live web research across role expectations, certifications, and industry standards
via DuckDuckGo HTTP search and optional Tavily API integration.
Operates defensively with timeout protection, source URL extraction, and honest status reporting.
"""
import os
import re
from typing import Any, Optional
from urllib.parse import unquote, urlparse
from pydantic import BaseModel, ConfigDict, Field
import httpx


class SearchResult(BaseModel):
    """Structured representation of a single live web research result."""
    model_config = ConfigDict(frozen=True)

    title: str = Field(..., description="Title of the web resource.")
    url: str = Field(..., description="Canonical destination URL.")
    snippet: str = Field(default="", description="Verbatim textual snippet from source.")
    domain: str = Field(default="", description="Hostname domain of source.")


class ResearchDossier(BaseModel):
    """Consolidated market and certification research dossier."""
    model_config = ConfigDict(frozen=True)

    query: str = Field(..., description="Executed research query.")
    provider: str = Field(..., description="Provider used ('duckduckgo', 'tavily', or 'offline_fallback').")
    status: str = Field(..., description="Status ('LIVE_SUCCESS', 'PARTIAL', 'OFFLINE', 'ERROR').")
    results: list[SearchResult] = Field(default_factory=list, description="Verified research citations.")
    summary: str = Field(default="", description="Synthesized takeaway or guidance.")
    is_live_research: bool = Field(default=False, description="Whether live external research succeeded.")


class WebResearcher:
    """
    Adapter providing live web intelligence for career pathways, certifications, and market expectations.
    Uses DuckDuckGo HTTP search (no API key required) and optional Tavily API if configured.
    """

    def __init__(self, timeout: float = 8.0) -> None:
        self.timeout = timeout
        self.tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip() or None

    def search(self, query: str, max_results: int = 5) -> ResearchDossier:
        """
        Execute live web query and return structured research dossier.
        Gracefully handles offline environments and network errors without raising exceptions.
        """
        clean_query = query.strip()
        if not clean_query:
            return ResearchDossier(
                query=query,
                provider="none",
                status="OFFLINE",
                results=[],
                summary="No search query provided.",
                is_live_research=False,
            )

        # 1. Try Tavily API if key configured
        if self.tavily_api_key:
            try:
                dossier = self._search_tavily(clean_query, max_results)
                if dossier.results:
                    return dossier
            except Exception:
                pass  # Fallback to DuckDuckGo

        # 2. Try DuckDuckGo HTTP search
        try:
            return self._search_duckduckgo(clean_query, max_results)
        except Exception as e:
            return ResearchDossier(
                query=clean_query,
                provider="duckduckgo",
                status="OFFLINE",
                results=[],
                summary=f"Live research temporarily unreachable: {type(e).__name__}. Utilizing verified model knowledge.",
                is_live_research=False,
            )

    def _search_duckduckgo(self, query: str, max_results: int = 5) -> ResearchDossier:
        """Query DuckDuckGo HTML endpoint and parse verified links."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        url = "https://html.duckduckgo.com/html/"
        resp = httpx.post(url, data={"q": query}, headers=headers, timeout=self.timeout)

        if resp.status_code != 200:
            return ResearchDossier(
                query=query,
                provider="duckduckgo",
                status="OFFLINE",
                results=[],
                summary=f"Search service responded with status {resp.status_code}.",
                is_live_research=False,
            )

        html = resp.text
        urls = re.findall(r'<a[^>]+class="[^"]*result__url[^"]*"[^>]*href="([^"]+)"[^>]*>', html)
        snippets = re.findall(r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>([\s\S]*?)</a>', html)
        titles = re.findall(r'<a[^>]+class="[^"]*result__a[^"]*"[^>]*>([\s\S]*?)</a>', html)

        items: list[SearchResult] = []
        count = min(len(urls), len(snippets), len(titles), max_results)

        for i in range(count):
            raw_url = urls[i]
            if "uddg=" in raw_url:
                m = re.search(r'uddg=([^&]+)', raw_url)
                if m:
                    raw_url = unquote(m.group(1))

            clean_title = re.sub(r'<[^>]+>', '', titles[i]).strip()
            clean_snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip()
            domain = urlparse(raw_url).netloc

            if clean_title and raw_url.startswith("http"):
                items.append(SearchResult(
                    title=clean_title,
                    url=raw_url,
                    snippet=clean_snippet,
                    domain=domain,
                ))

        if items:
            return ResearchDossier(
                query=query,
                provider="duckduckgo",
                status="LIVE_SUCCESS",
                results=items,
                summary=f"Retrieved {len(items)} live market sources for '{query}'.",
                is_live_research=True,
            )
        else:
            return ResearchDossier(
                query=query,
                provider="duckduckgo",
                status="PARTIAL",
                results=[],
                summary=f"No direct search matches returned for '{query}'.",
                is_live_research=False,
            )

    def _search_tavily(self, query: str, max_results: int = 5) -> ResearchDossier:
        """Query Tavily API if configured."""
        payload = {
            "api_key": self.tavily_api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        }
        resp = httpx.post("https://api.tavily.com/search", json=payload, timeout=self.timeout)
        if resp.status_code == 200:
            data = resp.json()
            items = []
            for r in data.get("results", [])[:max_results]:
                u = r.get("url", "")
                items.append(SearchResult(
                    title=r.get("title", "Market Intelligence Source"),
                    url=u,
                    snippet=r.get("content", ""),
                    domain=urlparse(u).netloc,
                ))
            return ResearchDossier(
                query=query,
                provider="tavily",
                status="LIVE_SUCCESS",
                results=items,
                summary=f"Retrieved {len(items)} live market sources from Tavily.",
                is_live_research=True,
            )
        raise RuntimeError(f"Tavily returned {resp.status_code}")

    def research_role_requirements(self, role_title: str) -> ResearchDossier:
        """Research real-world requirements, licensing, and certifications for any role."""
        role_lower = role_title.lower()
        if "insurance" in role_lower:
            query = f"{role_title} certification licensing IRDAI IC-38 requirements"
        elif any(w in role_lower for w in ["home loan", "mortgage", "loan advisor", "credit appraisal", "retail lending", "dsa"]):
            query = f"{role_title} retail lending DSA credit appraisal RBI guidelines"
        elif any(w in role_lower for w in ["home advisor", "property", "real estate", "realtor", "home buying", "housing"]):
            query = f"{role_title} property consulting RERA agent registration career skills"
        elif any(w in role_lower for w in ["securities", "mutual fund", "investment adviser", "wealth manager", "equity analyst", "portfolio manager", "stock broker"]):
            query = f"{role_title} certifications NISM CFP SEBI regulations"
        elif any(w in role_lower for w in ["banking", "credit analyst", "underwriting"]):
            query = f"{role_title} credit appraisal financial analysis certifications IIBF"
        elif any(w in role_lower for w in ["hr", "human resources", "talent", "recruiting"]):
            query = f"{role_title} certifications SHRM PHR competencies"
        elif any(w in role_lower for w in ["marketing", "sales", "business development"]):
            query = f"{role_title} key competencies certifications methodology"
        elif any(w in role_lower for w in ["data analyst", "bi analyst", "analytics", "data engineer"]):
            query = f"{role_title} SQL data warehousing certifications industry standards"
        elif any(w in role_lower for w in ["software", "ai", "machine learning", "devops", "cloud"]):
            query = f"{role_title} cloud microservices architecture certifications"
        else:
            query = f"{role_title} career role responsibilities competencies certifications"

        return self.search(query, max_results=4)

