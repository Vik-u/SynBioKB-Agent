from __future__ import annotations

from pathlib import Path
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib import colors


def h(doc, text, size=16):
    styles = getSampleStyleSheet()
    style = styles['Heading2']
    style.fontSize = size
    return Paragraph(text, style)


def p(text):
    styles = getSampleStyleSheet()
    return Paragraph(text, styles['BodyText'])


def code(text):
    styles = getSampleStyleSheet()
    s = styles['Code']
    return Paragraph(text.replace(' ', '&nbsp;'), s)


def bullet(items):
    styles = getSampleStyleSheet()
    return ListFlowable([ListItem(p(i)) for i in items], bulletType='bullet')


def build_pdf(out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out_path), pagesize=LETTER, title="Biosyn‑KB Setup & Usage Guide")
    content = []

    content.append(h(doc, "Biosyn‑KB: End‑to‑End Setup & Usage Guide", 18))
    content.append(Spacer(1, 0.25*inch))
    content.append(p("This document describes how to run the multi‑agent biosynthesis knowledge‑base pipeline and the web chat UI on your machine."))

    # Prerequisites
    content.append(Spacer(1, 0.2*inch))
    content.append(h(doc, "Prerequisites"))
    content.append(bullet([
        "macOS with Homebrew (or Linux with Python 3.11)",
        "SerpAPI API key (or Brave key) in apis.yaml",
        "Ollama installed locally with a model (e.g., gpt-oss:20b)",
    ]))

    content.append(Spacer(1, 0.2*inch))
    content.append(h(doc, "Project Layout"))
    content.append(p("All files are under the SynBioKB-Agent/ folder. Artifacts (DB/JSONL), pages, and work directories are stored there."))

    # Install
    content.append(Spacer(1, 0.2*inch))
    content.append(h(doc, "Install & Environment"))
    content.append(p("Create Python 3.11 environment, install project and CrewAI (this installs web UI deps too):"))
    content.append(code("""
cd biosyn-kb
make crewai
"""))

    # Config
    content.append(Spacer(1, 0.2*inch))
    content.append(h(doc, "Configuration"))
    content.append(p("Set search keys and LLM defaults: copy examples and fill values."))
    content.append(code("""
cp apis.yaml.example apis.yaml   # set SERPAPI_API_KEY or BRAVE_API_KEY
cp llm.yaml.example llm.yaml     # optional: set provider/model/base_url/temp/top_p/seed
"""))

    # Web app
    content.append(Spacer(1, 0.2*inch))
    content.append(h(doc, "Web Chat UI (preferred)"))
    content.append(p("Launch the web interface and open http://localhost:8000/chat:"))
    content.append(code("""
.venv311/bin/biosyn-kb-web
"""))
    content.append(p("In the chat UI: enter a query (e.g., 'biosynthesis of artemisinin'), choose provider/model, submit. A primer appears first, then structured evidence populates as background jobs finish. View job history at /status."))

    # CLI agent-query
    content.append(Spacer(1, 0.2*inch))
    content.append(h(doc, "One‑shot Agent (CLI)"))
    content.append(p("Run the full pipeline from a conversational query with minimal flags:"))
    content.append(code("""
.venv311/bin/biosyn-kb agent-query \
  --query "biosynthesis of artemisinin" \
  --provider serpapi --max-results 3 --exclude-domain wikipedia.org \
  --api-config apis.yaml \
  --save-html-dir pages/artemisinin --work-dir work/artemisinin \
  --db artifacts/artemisinin.db --out artifacts/artemisinin_summaries.jsonl \
  --limit 2
"""))
    content.append(p("Outputs: pages saved under pages/artemisinin/, JSONL summaries in artifacts/, and SQLite DB with pages and metrics."))

    # QA
    content.append(Spacer(1, 0.2*inch))
    content.append(h(doc, "Q&A (RAG over stored pages)"))
    content.append(code("""
.venv311/bin/biosyn-kb qa \
  --db artifacts/artemisinin.db \
  --question "What organism and enzymes are key in artemisinin biosynthesis and what titers are reported?" \
  --k 5 --model "gpt-oss:20b"
"""))

    # Queue
    content.append(Spacer(1, 0.2*inch))
    content.append(h(doc, "Queue & Worker (scaling)"))
    content.append(p("Enqueue an agent job and watch the worker process it:"))
    content.append(code("""
.venv311/bin/biosyn-kb queue-add-agent \
  --queue-db artifacts/queue.db \
  --query "biosynthesis of isobutanol" \
  --provider serpapi --api-config apis.yaml \
  --max-results 3 --exclude-domain wikipedia.org \
  --save-html-dir pages/agent

.venv311/bin/biosyn-kb queue-worker-watch --queue-db artifacts/queue.db --sleep 2
"""))

    # Tests
    content.append(Spacer(1, 0.2*inch))
    content.append(h(doc, "Run Unit Tests"))
    content.append(code("""
make test
"""))

    # Troubleshooting
    content.append(Spacer(1, 0.2*inch))
    content.append(h(doc, "Troubleshooting"))
    content.append(bullet([
        "Missing keys: set SERPAPI_API_KEY (or BRAVE_API_KEY) in apis.yaml",
        "Slow first QA: ONNX MiniLM downloads on first build; retry QA",
        "LLM slow on long pages: jobs run in background; use queue-worker-watch",
        "Forms error: install python-multipart (make crewai already installs it)",
    ]))

    doc.build(content)


if __name__ == "__main__":
    build_pdf(Path("docs/BiosynKB-Setup-Guide.pdf"))
