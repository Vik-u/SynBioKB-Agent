# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps that help build and run common libs (onnxruntime, chromadb, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install package and optional LLM agent deps
RUN python -m pip install -U pip \
 && pip install -e . \
 && pip install crewai langchain-community langchain-ollama \
 && pip install playwright \
 && python -m playwright install --with-deps chromium

# Expose FastAPI web
EXPOSE 8000

# Default environment (override as needed)
ENV LLM_PROVIDER=ollama \
    LLM_BASE_URL=http://host.docker.internal:11434

# Run the web app (includes API endpoints) — listens on :8000
CMD ["biosyn-kb-web"]
