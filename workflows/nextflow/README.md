# Nextflow Skeleton (Optional)

This folder contains a Nextflow skeleton that maps each stage of the pipeline to a process calling the `biosyn-kb` CLI. It is useful for HPC or cloud scale orchestration.

- main.nf: defines processes for `search`, `crawl`, `extract`, `summarize`, `store`, `pdfs`.
- nextflow.config: basic config; adjust executors as needed.

Note: Running Nextflow requires Java + Nextflow installed. This skeleton is provided for future use; the Python CLI already orchestrates the local demo.

