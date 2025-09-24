#!/usr/bin/env nextflow

params.query = params.query ?: 'biosynthesis of isobutanol'
params.max_results = params.max_results ?: 5
params.provider = params.provider ?: 'serpapi'
params.html_dir = params.html_dir ?: 'pages/nf'
params.out_jsonl = params.out_jsonl ?: 'summaries_nf.jsonl'
params.db = params.db ?: 'biosyn_nf.db'
params.model = params.model ?: 'gpt-oss:20b'

process search {
  output:
  path 'urls.txt'
  script:
  """
  mkdir -p ${params.html_dir}
  biosyn-kb search --query "${params.query}" --max-results ${params.max_results} --provider ${params.provider} \
    --exclude-domain wikipedia.org > urls.jsonl
  cat urls.jsonl | jq -r '.url' > urls.txt
  """
}

process crawl {
  input:
  path urls from search.out
  output:
  path params.html_dir
  script:
  """
  biosyn-kb crawl --urls-file ${urls} --save-html-dir ${params.html_dir}
  """
}

process extract {
  input:
  path params.html_dir from crawl.out
  output:
  path 'pages.jsonl'
  script:
  """
  biosyn-kb extract --html-dir ${params.html_dir} --out pages.jsonl --clean
  """
}

process summarize {
  input:
  path 'pages.jsonl' from extract.out
  output:
  path params.out_jsonl
  script:
  """
  biosyn-kb summarize --in pages.jsonl --out ${params.out_jsonl} --model "${params.model}" --concurrency 1
  """
}

process store {
  input:
  path params.out_jsonl from summarize.out
  output:
  path params.db
  script:
  """
  biosyn-kb db-init --db ${params.db}
  biosyn-kb db-import-summaries --db ${params.db} --in ${params.out_jsonl}
  """
}

workflow {
  take: query = params.query
  main:
    search()
    crawl()
    extract()
    summarize()
    store()
}

