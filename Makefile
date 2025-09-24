PY311=/opt/homebrew/bin/python3.11
VENV=.venv311
PIP=$(VENV)/bin/pip
PY=$(VENV)/bin/python
CLI=$(VENV)/bin/biosyn-kb

.PHONY: venv install crewai demo crewai-run qa test

venv:
	$(PY311) -m venv $(VENV)
	$(PY) -m pip install -U pip

install: venv
	$(PIP) install -e .

crewai: install
	$(PIP) install crewai langchain-community langchain-ollama

demo: install
	$(CLI) pipeline-run --html-dir pages --work-dir work/demo --db artifacts/biosyn_pipeline_demo.db --out artifacts/summaries_pipeline_demo.jsonl --model "gpt-oss:20b" --limit 2 --temperature 0.0 --top-p 1.0 --seed 42

crewai-run: crewai
	$(CLI) crewai-run --html-dir pages --work-dir work/crew --db artifacts/biosyn_crew311.db --out artifacts/summaries_crew311.jsonl --model "gpt-oss:20b" --limit 1 --temperature 0.0 --top-p 1.0 --seed 42

qa: install
	$(CLI) qa --db artifacts/biosyn_pipeline_demo.db --question "What are the yield and titer achieved for isobutanol in Nature Communications?" --k 5 --model "gpt-oss:20b" --temperature 0.0 --top-p 1.0 --seed 42

test: install
	$(PIP) install -q pytest
	$(VENV)/bin/pytest -q

.PHONY: docs
docs: install
	$(PIP) install -q reportlab
	$(PY) scripts/make_docs.py
	@echo "PDF written to docs/BiosynKB-Setup-Guide.pdf"

.PHONY: clean clean-artifacts compose-up compose-down demo-fast api-e2e
clean:
	@echo "Cleaning caches and build leftovers"
	@find . -name '__pycache__' -type d -prune -exec rm -rf {} + || true
	@find . -name '.pytest_cache' -type d -prune -exec rm -rf {} + || true
	@find . -name '.DS_Store' -type f -delete || true
	@rm -rf src/biosyn_kb/*.egg-info 2>/dev/null || true

clean-artifacts:
	@echo "Removing project outputs (pages/work/artifacts)"
	@rm -rf pages/* work/* artifacts/* 2>/dev/null || true

compose-up:
	@echo "Starting biosyn-kb via docker-compose"
	UNPAYWALL_EMAIL=$${UNPAYWALL_EMAIL:-you@example.com} docker compose up -d --build

compose-down:
	@docker compose down --remove-orphans || true

demo-fast: install
	@echo "Running fast demo (extract only, no LLM)"
	$(CLI) extract --html-dir pages --out artifacts/pages_clean.jsonl --clean --include-pdfs || true

api-e2e:
	@echo "Queueing E2E job against local API"
	curl -sS -X POST http://localhost:8000/api/e2e -H 'Content-Type: application/json' -d '{"query":"3-HP biosynthesis","api_config":"/app/apis.yaml","provider":"ollama","base_url":"http://host.docker.internal:11434","model":"gpt-oss:20b","limit":1,"max_results":5,"max_chars":8000}'
