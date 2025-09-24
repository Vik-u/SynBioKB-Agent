from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn

from ..queue import QueueDB, enqueue_job, run_worker_once_async
from ..llm import list_ollama_models
from ..llm.ollama_client import OllamaClient
from ..rag.vector_store import build_or_update_index, query_index
from ..store import init_db, import_pages_from_html_dir
import json
import uuid


app = FastAPI(title="Biosyn-KB Web")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
QUEUE_DB = Path("artifacts/web_queue.db")


@app.on_event("startup")
async def startup() -> None:
    # background worker
    app.state.worker_task = asyncio.create_task(worker_loop())


@app.on_event("shutdown")
async def shutdown() -> None:
    task: asyncio.Task = app.state.worker_task
    task.cancel()
    try:
        await task
    except Exception:
        pass


async def worker_loop(sleep: float = 2.0) -> None:
    q = QueueDB(QUEUE_DB)
    while True:
        try:
            job_id = await run_worker_once_async(q)
            if job_id is None:
                await asyncio.sleep(sleep)
            else:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(sleep)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    try:
        models = await list_ollama_models()
    except Exception:
        models = ["gpt-oss:20b"]
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "models": models,
        },
    )


@app.post("/run")
async def run(request: Request, query: str = Form(...), provider: str = Form("serpapi"), max_results: int = Form(5), exclude_domains: str = Form(""), model: str = Form("gpt-oss:20b")):
    # Enqueue agent (search + crawl), then crew pipeline
    ex = [d.strip() for d in exclude_domains.splitlines() if d.strip()]
    html_dir = "pages/web"
    work_dir = "work/web"
    db_path = "artifacts/web.db"
    out_jsonl = "artifacts/web_summaries.jsonl"
    q = QueueDB(QUEUE_DB)
    agent_id = enqueue_job(
        q,
        "agent",
        {
            "query": query,
            "provider": provider,
            "api_config_path": "apis.yaml",
            "max_results": max_results,
            "exclude_domains": ex,
            "save_html_dir": html_dir,
        },
    )
    crew_id = enqueue_job(
        q,
        "crew",
        {
            "html_dir": html_dir,
            "work_dir": work_dir,
            "db_path": db_path,
            "out_jsonl": out_jsonl,
            "model": model,
            "ollama_url": "http://localhost:11434",
            "limit": 3,
            "max_chars": 6000,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
        },
    )
    return RedirectResponse(url="/status", status_code=303)


# -------- JSON API --------

@app.post("/api/e2e")
async def api_e2e(request: Request):
    body = await request.json()
    query = body.get("query") or ""
    if not query:
        return JSONResponse({"error": "query is required"}, status_code=400)
    provider = body.get("provider") or "ollama"
    base_url = body.get("base_url") or "http://localhost:11434"
    api_key = body.get("api_key")
    model = body.get("model") or "gpt-oss:20b"
    api_config_path = body.get("api_config") or "apis.yaml"
    limit = int(body.get("limit") or 5)
    max_results = int(body.get("max_results") or 50)
    max_chars = int(body.get("max_chars") or 12000)
    temperature = float(body.get("temperature") or 0.0)
    top_p = float(body.get("top_p") or 1.0)
    seed = int(body.get("seed") or 42)

    html_dir = body.get("html_dir") or "pages/api_e2e"
    work_dir = body.get("work_dir") or "work/api_e2e"
    db_path = body.get("db_path") or "artifacts/api_e2e.db"
    out_jsonl = body.get("out_jsonl") or "artifacts/api_e2e_summaries.jsonl"

    q = QueueDB(QUEUE_DB)
    job_id = enqueue_job(
        q,
        "e2e",
        {
            "query": query,
            "provider": provider,
            "base_url": base_url,
            "api_key": api_key,
            "api_config_path": api_config_path,
            "model": model,
            "limit": limit,
            "max_results": max_results,
            "max_chars": max_chars,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "html_dir": html_dir,
            "work_dir": work_dir,
            "db_path": db_path,
            "out_jsonl": out_jsonl,
        },
    )
    return JSONResponse({"status": "queued", "job_id": job_id})


@app.get("/api/jobs")
async def api_jobs(limit: int = 50):
    q = QueueDB(QUEUE_DB)
    q.init()
    conn = q.connect()
    rows = conn.execute("SELECT id,type,status,error,created_at,started_at,finished_at FROM jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    data = [dict(id=r[0], type=r[1], status=r[2], error=r[3], created_at=r[4], started_at=r[5], finished_at=r[6]) for r in rows]
    return JSONResponse(data)


@app.get("/api/job/{job_id}")
async def api_job(job_id: int):
    q = QueueDB(QUEUE_DB)
    q.init()
    conn = q.connect()
    row = conn.execute("SELECT id,type,status,error,created_at,started_at,finished_at FROM jobs WHERE id=?", (job_id,)).fetchone()
    conn.close()
    if not row:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(dict(id=row[0], type=row[1], status=row[2], error=row[3], created_at=row[4], started_at=row[5], finished_at=row[6]))


@app.get("/status", response_class=HTMLResponse)
async def status(request: Request):
    # Show last 50 jobs
    q = QueueDB(QUEUE_DB)
    q.init()
    conn = q.connect()
    rows = conn.execute("SELECT id,type,status,error,created_at,started_at,finished_at FROM jobs ORDER BY id DESC LIMIT 50").fetchall()
    conn.close()
    return templates.TemplateResponse("status.html", {"request": request, "rows": rows})


# ---------------- Chat interface -----------------


def _session_dir() -> Path:
    d = Path("artifacts/chat_sessions")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _session_path(sid: str) -> Path:
    return _session_dir() / f"{sid}.jsonl"


def _append_chat(sid: str, role: str, text: str) -> None:
    rec = {"role": role, "text": text}
    p = _session_path(sid)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _load_chat(sid: str) -> list[dict]:
    p = _session_path(sid)
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _build_reply_from_summaries(path: Path) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return "No structured summaries available yet."
    lines = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    parts: list[str] = []
    for obj in lines[:3]:
        title = obj.get("title") or obj.get("url")
        url = obj.get("url")
        chem = obj.get("chemical")
        orgs = ", ".join(o.get("name") for o in (obj.get("organisms") or []) if o.get("name"))
        enzymes = ", ".join(e.get("name") for e in (obj.get("enzymes") or []) if e.get("name"))
        metrics = "; ".join(f"{m.get('kind')}: {m.get('value')} {m.get('unit') or ''}" for m in (obj.get("metrics") or []))
        parts.append(f"<b>Source:</b> <a href='{url}' target='_blank'>{title}</a><br><b>Chemical:</b> {chem or ''}<br><b>Organisms:</b> {orgs or ''}<br><b>Enzymes:</b> {enzymes or ''}<br><b>Metrics:</b> {metrics or ''}")
    return "<hr>".join(parts) if parts else "No structured summaries parsed."


@app.get("/chat", response_class=HTMLResponse)
async def chat_ui(request: Request, session: Optional[str] = None):
    sid = session or uuid.uuid4().hex[:8]
    msgs = _load_chat(sid)
    try:
        models = await list_ollama_models()
    except Exception:
        models = ["gpt-oss:20b"]
    # If summaries exist for this session, build a reply block
    out_jsonl = Path(f"artifacts/chat_{sid}_summaries.jsonl")
    reply_html = _build_reply_from_summaries(out_jsonl)
    return templates.TemplateResponse(
        "chat.html",
        {"request": request, "session": sid, "messages": msgs, "models": models, "reply_html": reply_html},
    )


@app.post("/chat")
async def chat_send(
    request: Request,
    message: str = Form(...),
    model: str = Form("gpt-oss:20b"),
    provider: str = Form("serpapi"),
    max_results: int = Form(3),
    exclude_domains: str = Form("wikipedia.org"),
    session: str = Form(None),
):
    sid = session or uuid.uuid4().hex[:8]
    _append_chat(sid, "user", message)
    # Quick primer
    primer_prompt = (
        "You are a biosynthesis expert. Provide a concise primer for the following topic,"
        " covering pathways, key enzymes (with EC numbers when known), likely organisms,"
        " feedstocks, and typical performance ranges (yield/titer/productivity)."
        " Use cautious language and 4-6 bullets. If uncertain, say so.\n\n"
        f"Topic: {message}\n\nPrimer:"
    )
    try:
        async with OllamaClient() as oc:
            primer = await oc.generate(model, primer_prompt, options={"temperature": 0.2, "top_p": 1.0})
        _append_chat(sid, "assistant", primer)
    except Exception:
        pass
    # Enqueue agent + crew for this session
    q = QueueDB(QUEUE_DB)
    ex = [d.strip() for d in exclude_domains.splitlines() if d.strip()]
    html_dir = f"pages/chat/{sid}"
    work_dir = f"work/chat/{sid}"
    db_path = f"artifacts/chat_{sid}.db"
    out_jsonl = f"artifacts/chat_{sid}_summaries.jsonl"
    enqueue_job(
        q,
        "agent",
        {
            "query": message,
            "provider": provider,
            "api_config_path": "apis.yaml",
            "max_results": max_results,
            "exclude_domains": ex,
            "save_html_dir": html_dir,
        },
    )
    enqueue_job(
        q,
        "crew",
        {
            "html_dir": html_dir,
            "work_dir": work_dir,
            "db_path": db_path,
            "out_jsonl": out_jsonl,
            "model": model,
            "ollama_url": "http://localhost:11434",
            "limit": 3,
            "max_chars": 6000,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
        },
    )
    return RedirectResponse(url=f"/chat?session={sid}", status_code=303)


def run():
    uvicorn.run("biosyn_kb.webapp.main:app", host="0.0.0.0", port=8000, reload=False)
