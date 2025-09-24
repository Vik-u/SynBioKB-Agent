from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import chromadb
from chromadb.api.models.Collection import Collection


def _tokenize(text: str) -> List[str]:
    import re

    return [w.lower() for w in re.findall(r"[a-zA-Z0-9-]+", text or "") if len(w) > 2]


@dataclass
class RAGIndex:
    client: chromadb.Client
    collection: Collection


def build_or_update_index(db_path: str | Path, persist_dir: str | Path = "artifacts/rag") -> RAGIndex:
    from ..store.db import _engine  # reuse engine
    from sqlalchemy.orm import Session
    from sqlalchemy import select
    from ..store.db import Page
    from chromadb.utils import embedding_functions

    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    # Always require SentenceTransformers for embeddings; no hashing fallback
    try:
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    except Exception as e:
        raise RuntimeError(
            "SentenceTransformers embedding is required. Install 'sentence-transformers' and retry."
        ) from e

    coll = client.get_or_create_collection(name="pages_paragraphs", embedding_function=embed_fn)  # type: ignore[arg-type]

    # Fetch paragraphs and upsert
    ids: List[str] = []
    docs: List[str] = []
    metas: List[dict] = []

    engine = _engine(db_path)
    with Session(engine) as ses:
        pages = ses.execute(select(Page)).scalars().all()
        for i, pg in enumerate(pages):
            text = pg.cleaned_text or ""
            if not text:
                continue
            import re

            paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
            for j, para in enumerate(paras):
                pid = f"{i}:{j}:{hash(para) & ((1<<32)-1)}"
                ids.append(pid)
                docs.append(para)
                metas.append({"url": pg.url, "title": pg.title or ""})

    if docs:
        # Upsert in batches
        B = 512
        for k in range(0, len(docs), B):
            coll.upsert(ids=ids[k : k + B], documents=docs[k : k + B], metadatas=metas[k : k + B])

    return RAGIndex(client=client, collection=coll)


def query_index(persist_dir: str | Path, query: str, k: int = 5) -> List[dict]:
    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        coll = client.get_collection(name="pages_paragraphs")
    except Exception:
        return []
    res = coll.query(query_texts=[query], n_results=k)
    outs: List[dict] = []
    docs = res.get("documents") or [[]]
    metas = res.get("metadatas") or [[]]
    for d, m in zip(docs[0], metas[0]):
        outs.append({"para": d, "url": m.get("url"), "title": m.get("title")})
    return outs
