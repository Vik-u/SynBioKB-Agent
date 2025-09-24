from __future__ import annotations

from typing import List, Optional
import re
import httpx

from ..api_config import ApiConfig


_DOI_RX = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")


def guess_doi_from_url(url: str) -> Optional[str]:
    m = _DOI_RX.search(url)
    return m.group(0) if m else None


async def unpaywall_oa_locations(doi: str, email: Optional[str]) -> List[str]:
    if not doi or not email:
        return []
    u = f"https://api.unpaywall.org/v2/{doi}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(u, params={"email": email})
            r.raise_for_status()
            js = r.json()
            urls: List[str] = []
            # Prefer best_oa_location
            best = js.get("best_oa_location") or {}
            for k in ("url", "url_for_pdf", "url_for_landing_page"):
                v = best.get(k)
                if v:
                    urls.append(v)
            # Also collect other OA locations
            for loc in js.get("oa_locations") or []:
                for k in ("url", "url_for_pdf", "url_for_landing_page"):
                    v = loc.get(k)
                    if v and v not in urls:
                        urls.append(v)
            # Prefer PMC HTML if present
            urls.sort(key=lambda x: ("ncbi.nlm.nih.gov/pmc" not in x, "/pdf" in x))
            return urls
    except Exception:
        return []


async def enrich_urls_with_oa(urls: List[str], api_cfg: ApiConfig) -> List[str]:
    # For URLs that contain a DOI, query Unpaywall for OA alternatives
    extra: List[str] = []
    email = api_cfg.unpaywall.email if api_cfg and api_cfg.unpaywall else None
    if not email:
        return urls  # no OA enrichment without contact email
    seen = set(urls)
    for u in urls:
        doi = guess_doi_from_url(u)
        if not doi:
            continue
        locs = await unpaywall_oa_locations(doi, email)
        for v in locs:
            if v not in seen:
                extra.append(v)
                seen.add(v)
    return urls + extra

