from __future__ import annotations

from typing import Optional, Tuple

import httpx


async def pubchem_inchikey_for_name(name: str, *, timeout: float = 8.0) -> Tuple[Optional[str], Optional[int]]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{httpx.utils.quote(name)}/cids/JSON")
            r.raise_for_status()
            cids = r.json().get("IdentifierList", {}).get("CID", [])
            if not cids:
                return None, None
            cid = cids[0]
            r2 = await client.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/InChIKey/JSON")
            r2.raise_for_status()
            props = r2.json().get("PropertyTable", {}).get("Properties", [])
            if props:
                return props[0].get("InChIKey"), cid
    except Exception:
        return None, None
    return None, None


async def uniprot_for_ec(ec: str, *, timeout: float = 8.0) -> Optional[str]:
    # Return a single UniProt accession for the EC as a coarse normalization
    try:
        q = f"ec:{ec}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get("https://rest.uniprot.org/uniprotkb/search", params={"query": q, "size": 1, "fields": "accession"})
            r.raise_for_status()
            js = r.json()
            if js.get("results"):
                return js["results"][0]["primaryAccession"]
    except Exception:
        return None
    return None


async def taxonomy_id_for_name(name: str, *, timeout: float = 8.0) -> Optional[int]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"https://www.ebi.ac.uk/ena/taxonomy/rest/scientific-name/{httpx.utils.quote(name)}")
            if r.status_code == 200:
                js = r.json()
                if isinstance(js, list) and js:
                    tid = js[0].get("taxId")
                    return int(tid) if tid else None
    except Exception:
        return None
    return None

