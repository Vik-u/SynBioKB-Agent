// Nextflow orchestration for Biosyn-KB
// Supports two modes:
//  - e2e=true  : single LLM-first end-to-end pipeline (strategist → OA search → crawl → extract → summarize → store → compose)
//  - e2e=false : modular steps (search/crawl → extract → summarize → store → compose)

nextflow.enable.dsl=2

workflow {
  if (params.e2e) {
    CREWAI_E2E()
  } else {
    MODULAR()
  }
}

// -------------------- E2E (LLM-first) --------------------

process CREWAI_E2E {
  tag "e2e"
  cpus 1
  memory '4 GB'
  errorStrategy 'terminate'
  publishDir params.outdir, mode: 'copy', overwrite: true

  input:
  val QUERY from Channel.value(params.query)

  output:
  path 'report.md', emit: report
  path 'summaries.jsonl', emit: summaries
  path 'kb.sqlite', emit: db
  path 'checkpoints', optional: true, emit: checkpoints

  script:
  def apiKeyArg = params.llm_api_key ? "--api-key ${params.llm_api_key}" : ""
  def apiCfgArg = params.api_config ? "--api-config ${params.api_config}" : ""
  def baseUrlArg = params.llm_base_url ?: 'http://localhost:11434'
  def pagesDir = params.pagesdir ?: "${params.outdir}/pages"
  def workDir  = params.workdir  ?: "${params.outdir}/work"
  def dbPath   = params.db       ?: "${params.outdir}/kb.sqlite"
  def outJsonl = params.summaries?: "${params.outdir}/summaries.jsonl"
  """
  set -euo pipefail
  mkdir -p ${params.outdir}

  biosyn-kb crewai-e2e \
    --query "${params.query}" \
    ${apiCfgArg} \
    --work-dir "${workDir}" \
    --html-dir "${pagesDir}" \
    --db "${dbPath}" \
    --out "${outJsonl}" \
    --provider "${params.llm_provider}" \
    --base-url "${baseUrlArg}" \
    ${apiKeyArg} \
    --model "${params.model}" \
    --limit ${params.limit} \
    --max-results ${params.max_results} \
    --max-chars ${params.max_chars} \
    --temperature ${params.temperature} \
    --top-p ${params.top_p} \
    --seed ${params.seed}

  # Collect key outputs into publishDir
  cp "${workDir}/report.md" report.md
  cp "${outJsonl}" summaries.jsonl
  cp "${dbPath}" kb.sqlite
  if [ -d "${workDir}/checkpoints" ]; then
    cp -R "${workDir}/checkpoints" ./checkpoints
  fi
  """
}

// -------------------- Modular pipeline --------------------

process SEARCH_CRAWL {
  tag 'search_crawl'
  cpus 1
  memory '2 GB'
  publishDir params.pagesdir, mode: 'copy', overwrite: true

  input:
  val QUERY from Channel.value(params.query)

  output:
  path 'pages_done.txt'

  script:
  def apiCfgArg = params.api_config ? "--api-config ${params.api_config}" : ""
  def pagesDir = params.pagesdir ?: "${params.outdir}/pages"
  """
  set -euo pipefail
  mkdir -p ${pagesDir}
  biosyn-kb run-all \
    --query "${params.query}" \
    --max-results ${params.max_results} \
    --provider ${params.provider} \
    ${apiCfgArg} \
    --save-html-dir "${pagesDir}" \
    --per-domain-delay 1.0 \
    --max-concurrency 4 \
    > "${pagesDir}/crawl_meta.jsonl"
  echo done > pages_done.txt
  """
}

process EXTRACT {
  tag 'extract'
  cpus 1
  memory '2 GB'
  publishDir params.workdir, mode: 'copy', overwrite: true

  input:
  path _CRAWL from SEARCH_CRAWL.out

  output:
  path 'extracted.jsonl'

  script:
  def pagesDir = params.pagesdir ?: "${params.outdir}/pages"
  def workDir  = params.workdir  ?: "${params.outdir}/work"
  """
  set -euo pipefail
  mkdir -p ${workDir}
  biosyn-kb extract \
    --html-dir "${pagesDir}" \
    --out "${workDir}/extracted.jsonl" \
    --clean \
    --include-pdfs
  cp "${workDir}/extracted.jsonl" extracted.jsonl
  """
}

process SUMMARIZE {
  tag 'summarize'
  cpus 1
  memory '8 GB'
  publishDir params.outdir, mode: 'copy', overwrite: true

  input:
  path INP from EXTRACT.out

  output:
  path 'summaries.jsonl'

  script:
  def apiKeyArg = params.llm_api_key ? "--api-key ${params.llm_api_key}" : ""
  def baseUrlArg = params.llm_base_url ?: 'http://localhost:11434'
  def workDir  = params.workdir  ?: "${params.outdir}/work"
  def outJsonl = params.summaries?: "${params.outdir}/summaries.jsonl"
  """
  set -euo pipefail
  biosyn-kb summarize \
    --in "${INP}" \
    --out "${outJsonl}" \
    --provider "${params.llm_provider}" \
    --base-url "${baseUrlArg}" \
    ${apiKeyArg} \
    --model "${params.model}" \
    --limit ${params.limit} \
    --concurrency 1 \
    --max-chars ${params.max_chars} \
    --temperature ${params.temperature} \
    --top-p ${params.top_p} \
    --seed ${params.seed} \
    --chunked --chunk-chars 3500
  cp "${outJsonl}" summaries.jsonl
  """
}

process DB_IMPORT {
  tag 'db_import'
  cpus 1
  memory '2 GB'
  publishDir params.outdir, mode: 'copy', overwrite: true

  input:
  path SUMM from SUMMARIZE.out

  output:
  path 'kb.sqlite'

  script:
  def pagesDir = params.pagesdir ?: "${params.outdir}/pages"
  def dbPath   = params.db       ?: "${params.outdir}/kb.sqlite"
  """
  set -euo pipefail
  biosyn-kb db-init --db "${dbPath}"
  biosyn-kb db-import-pages --db "${dbPath}" --html-dir "${params.pagesdir}" --include-pdfs
  biosyn-kb db-import-summaries --db "${dbPath}" --in "${SUMM}"
  cp "${dbPath}" kb.sqlite
  """
}

process COMPOSE {
  tag 'compose'
  cpus 1
  memory '2 GB'
  publishDir params.outdir, mode: 'copy', overwrite: true

  input:
  path SUMM from SUMMARIZE.out

  output:
  path 'report.md'

  script:
  def apiKeyArg = params.llm_api_key ? "--api-key ${params.llm_api_key}" : ""
  def baseUrlArg = params.llm_base_url ?: 'http://localhost:11434'
  """
  set -euo pipefail
  biosyn-kb compose-report \
    --in "${SUMM}" \
    --out report.md \
    --provider "${params.llm_provider}" \
    --base-url "${baseUrlArg}" \
    ${apiKeyArg} \
    --model "${params.model}"
  """
}

workflow MODULAR {
  def q = Channel.value(params.query)
  def c = SEARCH_CRAWL(q)
  def e = EXTRACT(c)
  def s = SUMMARIZE(e)
  DB_IMPORT(s)
  COMPOSE(s)
}

