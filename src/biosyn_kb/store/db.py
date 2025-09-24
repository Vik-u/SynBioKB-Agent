from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

import re
from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    select,
    delete,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session


class Base(DeclarativeBase):
    pass


class Summary(Base):
    __tablename__ = "summaries"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    url: Mapped[str] = mapped_column(String(1000), unique=True, index=True)
    title: Mapped[Optional[str]] = mapped_column(String(1000))
    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    journal_or_source: Mapped[Optional[str]] = mapped_column(String(255))
    chemical: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    approach: Mapped[Optional[str]] = mapped_column(String(255))
    pathway: Mapped[Optional[str]] = mapped_column(String(255))
    conditions: Mapped[Optional[str]] = mapped_column(Text)
    raw_json: Mapped[str] = mapped_column(Text)

    metrics: Mapped[List["Metric"]] = relationship(back_populates="summary", cascade="all, delete-orphan")
    organisms: Mapped[List["Organism"]] = relationship(back_populates="summary", cascade="all, delete-orphan")
    enzymes: Mapped[List["Enzyme"]] = relationship(back_populates="summary", cascade="all, delete-orphan")
    feedstocks: Mapped[List["Feedstock"]] = relationship(back_populates="summary", cascade="all, delete-orphan")
    starting_substrates: Mapped[List["StartingSubstrate"]] = relationship(back_populates="summary", cascade="all, delete-orphan")
    reaction_steps: Mapped[List["ReactionStep"]] = relationship(back_populates="summary", cascade="all, delete-orphan")
    strain_mods: Mapped[List["StrainModification"]] = relationship(back_populates="summary", cascade="all, delete-orphan")


class Page(Base):
    __tablename__ = "pages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    url: Mapped[str] = mapped_column(String(1000), unique=True, index=True)
    title: Mapped[Optional[str]] = mapped_column(String(1000))
    content_type: Mapped[Optional[str]] = mapped_column(String(255))
    encoding: Mapped[Optional[str]] = mapped_column(String(64))
    raw_html: Mapped[Optional[str]] = mapped_column(Text)
    cleaned_text: Mapped[Optional[str]] = mapped_column(Text)
    content_hash: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    saved_path: Mapped[Optional[str]] = mapped_column(String(1000))
    pdf_path: Mapped[Optional[str]] = mapped_column(String(1000))
    notes: Mapped[Optional[str]] = mapped_column(Text)


class Metric(Base):
    __tablename__ = "metrics"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    summary_id: Mapped[int] = mapped_column(ForeignKey("summaries.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(50))
    value: Mapped[Optional[float]] = mapped_column(Float)
    unit: Mapped[Optional[str]] = mapped_column(String(50))
    conditions: Mapped[Optional[str]] = mapped_column(Text)
    summary: Mapped[Summary] = relationship(back_populates="metrics")


class Organism(Base):
    __tablename__ = "organisms"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    summary_id: Mapped[int] = mapped_column(ForeignKey("summaries.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    role: Mapped[Optional[str]] = mapped_column(String(255))
    summary: Mapped[Summary] = relationship(back_populates="organisms")


class Enzyme(Base):
    __tablename__ = "enzymes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    summary_id: Mapped[int] = mapped_column(ForeignKey("summaries.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    ec_number: Mapped[Optional[str]] = mapped_column(String(50))
    organism: Mapped[Optional[str]] = mapped_column(String(255))
    reaction_type: Mapped[Optional[str]] = mapped_column(String(255))
    enzyme_class: Mapped[Optional[str]] = mapped_column(String(255))
    engineered: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    mutations: Mapped[Optional[str]] = mapped_column(Text)
    summary: Mapped[Summary] = relationship(back_populates="enzymes")


class Feedstock(Base):
    __tablename__ = "feedstocks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    summary_id: Mapped[int] = mapped_column(ForeignKey("summaries.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    summary: Mapped[Summary] = relationship(back_populates="feedstocks")


class StartingSubstrate(Base):
    __tablename__ = "starting_substrates"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    summary_id: Mapped[int] = mapped_column(ForeignKey("summaries.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    summary: Mapped[Summary] = relationship(back_populates="starting_substrates")


class ReactionStep(Base):
    __tablename__ = "reaction_steps"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    summary_id: Mapped[int] = mapped_column(ForeignKey("summaries.id", ondelete="CASCADE"), index=True)
    substrate: Mapped[str] = mapped_column(String(255))
    product: Mapped[str] = mapped_column(String(255))
    enzyme_name: Mapped[Optional[str]] = mapped_column(String(255))
    ec_number: Mapped[Optional[str]] = mapped_column(String(50))
    organism: Mapped[Optional[str]] = mapped_column(String(255))
    reaction_type: Mapped[Optional[str]] = mapped_column(String(255))
    enzyme_class: Mapped[Optional[str]] = mapped_column(String(255))
    engineered: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    mutations: Mapped[Optional[str]] = mapped_column(Text)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    summary: Mapped[Summary] = relationship(back_populates="reaction_steps")


class StrainModification(Base):
    __tablename__ = "strain_mods"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    summary_id: Mapped[int] = mapped_column(ForeignKey("summaries.id", ondelete="CASCADE"), index=True)
    gene: Mapped[str] = mapped_column(String(255), index=True)
    action: Mapped[str] = mapped_column(String(64))
    details: Mapped[Optional[str]] = mapped_column(Text)
    summary: Mapped[Summary] = relationship(back_populates="strain_mods")


def _engine(db_path: str | Path):
    return create_engine(f"sqlite:///{db_path}", future=True)


def init_db(db_path: str | Path) -> None:
    engine = _engine(db_path)
    Base.metadata.create_all(engine)


def import_summaries_jsonl(db_path: str | Path, input_jsonl: str | Path) -> int:
    engine = _engine(db_path)
    created = 0
    with Session(engine) as ses:
        with open(input_jsonl, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                url = obj.get("url")
                if not url:
                    continue
                # Upsert: delete existing summary for URL
                ses.execute(delete(Summary).where(Summary.url == url))
                summ = Summary(
                    url=url,
                    title=obj.get("title"),
                    year=obj.get("year"),
                    journal_or_source=obj.get("journal_or_source"),
                    chemical=(obj.get("chemical") or None),
                    approach=obj.get("approach"),
                    pathway=obj.get("pathway"),
                    conditions=obj.get("conditions"),
                    raw_json=json.dumps(obj, ensure_ascii=False),
                )
                ses.add(summ)
                ses.flush()

                for m in obj.get("metrics", []) or []:
                    ses.add(Metric(summary_id=summ.id, kind=m.get("kind"), value=m.get("value"), unit=m.get("unit"), conditions=m.get("conditions")))
                for o in obj.get("organisms", []) or []:
                    ses.add(Organism(summary_id=summ.id, name=o.get("name"), role=o.get("role")))
                for e in obj.get("enzymes", []) or []:
                    ses.add(Enzyme(summary_id=summ.id, name=e.get("name"), ec_number=e.get("ec_number"), organism=e.get("organism"), reaction_type=e.get("reaction_type"), enzyme_class=e.get("enzyme_class"), engineered=e.get("engineered"), mutations=(None if e.get("mutations") is None else json.dumps(e.get("mutations")))))
                for s in obj.get("feedstocks", []) or []:
                    ses.add(Feedstock(summary_id=summ.id, name=s))
                for s in obj.get("starting_substrates", []) or []:
                    ses.add(StartingSubstrate(summary_id=summ.id, name=s))
                for rs in obj.get("reaction_steps", []) or []:
                    enz = rs.get("enzyme") or {}
                    ses.add(
                        ReactionStep(
                            summary_id=summ.id,
                            substrate=rs.get("substrate") or "",
                            product=rs.get("product") or "",
                            enzyme_name=enz.get("name"),
                            ec_number=enz.get("ec_number"),
                            organism=enz.get("organism"),
                            reaction_type=enz.get("reaction_type"),
                            enzyme_class=enz.get("enzyme_class"),
                            engineered=enz.get("engineered"),
                            mutations=(None if enz.get("mutations") is None else json.dumps(enz.get("mutations"))),
                            notes=rs.get("notes"),
                        )
                    )
                for sm in obj.get("strain_design", []) or []:
                    ses.add(StrainModification(summary_id=summ.id, gene=sm.get("gene") or "", action=sm.get("action") or "", details=sm.get("details")))
                created += 1
        ses.commit()
    return created


def import_pages_from_html_dir(db_path: str | Path, html_dir: str | Path, *, use_cleaner: bool = True, include_pdfs: bool = False, pdf_dir: str | Path | None = None) -> int:
    from ..extract.clean import extract_clean_text
    from ..extract.pdf import extract_pdf_text
    import hashlib
    engine = _engine(db_path)
    created = 0
    html_dir = Path(html_dir)
    with Session(engine) as ses:
        for path in sorted(html_dir.glob("*.html")):
            url_guess = str(path)
            html = path.read_text(encoding="utf-8", errors="ignore")
            cleaned = extract_clean_text(html, url=None) if use_cleaner else None
            title = (cleaned.title if cleaned else None)
            text = (cleaned.text if cleaned else None)
            h = hashlib.sha256((text or html).encode("utf-8", errors="ignore")).hexdigest()
            ses.add(
                Page(
                    url=url_guess,
                    title=title,
                    content_type="text/html",
                    encoding="utf-8",
                    raw_html=html,
                    cleaned_text=text,
                    content_hash=h,
                    saved_path=str(path),
                )
            )
            created += 1
        # Optionally import PDFs from the same directory or provided pdf_dir
        if include_pdfs:
            pdir = Path(pdf_dir) if pdf_dir else html_dir
            for p in sorted(pdir.glob("*.pdf")):
                try:
                    pdfc = extract_pdf_text(str(p))
                    text = pdfc.text
                    if not text:
                        continue
                    h = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
                    ses.add(
                        Page(
                            url=str(p),
                            title=pdfc.title,
                            content_type="application/pdf",
                            encoding=None,
                            raw_html=None,
                            cleaned_text=text,
                            content_hash=h,
                            saved_path=str(p),
                            pdf_path=str(p),
                        )
                    )
                    created += 1
                except Exception:
                    continue
        ses.commit()
    return created


def query_metrics(db_path: str | Path, chemical: Optional[str] = None, limit: int = 20) -> List[dict]:
    engine = _engine(db_path)
    rows: List[dict] = []
    with Session(engine) as ses:
        q = select(Summary, Metric).join(Metric, Metric.summary_id == Summary.id)
        if chemical:
            q = q.where(Summary.chemical == chemical)
        q = q.limit(limit)
        for s, m in ses.execute(q):
            rows.append(
                {
                    "url": s.url,
                    "title": s.title,
                    "chemical": s.chemical,
                    "kind": m.kind,
                    "value": m.value,
                    "unit": m.unit,
                    "conditions": m.conditions,
                }
            )
    return rows


def retrieve_paragraphs(db_path: str | Path, query: str, *, chemical: Optional[str] = None, k: int = 5) -> List[dict]:
    """Naive RAG retrieval: get top-k paragraphs matching tokens in query.

    Tokenize simple words, scan cleaned_text from pages, and pick paragraphs with highest hit count.
    """
    tokens = [t.lower() for t in re.findall(r"[a-zA-Z0-9-]+", query or "") if len(t) > 2]
    engine = _engine(db_path)
    from collections import Counter
    results: List[tuple[int, dict]] = []
    with Session(engine) as ses:
        pages = ses.execute(select(Page)).scalars().all()
        for pg in pages:
            text = pg.cleaned_text or ""
            paras = re.split(r"\n\s*\n+", text)
            for para in paras:
                p_low = para.lower()
                score = sum(p_low.count(t) for t in tokens)
                if score > 0:
                    results.append((score, {"url": pg.url, "title": pg.title, "para": para.strip()}))
    results.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in results[:k]]
