#!/usr/bin/env bash
set -euo pipefail

QUERY=${1:-"3-HP biosynthesis"}
OUT=${2:-"ab_test"}

TMP=$(mktemp)
echo "[Search] collecting 2 URLs for: $QUERY"
biosyn-kb search --query "$QUERY" --max-results 2 --provider serpapi --api-config apis.yaml | jq -r '.["url"]? // .url' 2>/dev/null | sed '/^$/d' > "$TMP"

echo "[A] httpx crawler"
biosyn-kb crawl --urls-file "$TMP" --save-html-dir pages/${OUT}_httpx --per-domain-delay 1.0 --max-concurrency 4 >/dev/null || true
CHTTP=$(ls -1 pages/${OUT}_httpx/*.html 2>/dev/null | wc -l | tr -d ' ')

echo "[B] headless crawler"
biosyn-kb crawl --urls-file "$TMP" --save-html-dir pages/${OUT}_headless --per-domain-delay 1.0 --max-concurrency 2 --headless >/dev/null || true
CHL=$(ls -1 pages/${OUT}_headless/*.html 2>/dev/null | wc -l | tr -d ' ')
rm -f "$TMP"

echo "Result: httpx=$CHTTP, headless=$CHL"
if [[ "$CHL" -gt "$CHTTP" ]]; then
  echo "Headless captured more HTML pages (likely JS-protected sites)."
else
  echo "HTTPX captured same or more pages (headless not required for this query)."
fi
