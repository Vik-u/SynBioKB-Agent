from __future__ import annotations

PROMPT_TEMPLATE = (
    "You are a scientific extraction assistant. Given a web page title and body text, "
    "extract a structured summary about biosynthesis/bioproduction facts. Keep it concise and factual.\n\n"
    "Return ONLY valid JSON matching this schema (no Markdown). Fields to fill when present: \n"
    "- url: string\n"
    "- title: string\n"
    "- year: integer (if found)\n"
    "- journal_or_source: string (journal name, conference, or site)\n"
    "- chemical: string\n"
    "- approach: string (e.g., microbial, cell-free, engineered pathway)\n"
    "- pathway: string (biosynthetic pathway name if given)\n"
    "- organisms: [{name, role?}]\n"
    "- enzymes: [{name, ec_number?}]\n"
    "- feedstocks: [string] (e.g., glucose, glycerol)\n"
    "- starting_substrates: [string] (initial precursors for pathway, if specified)\n"
    "- metrics: [{kind: 'yield'|'titer'|'productivity'|'selectivity'|'conversion', value: number, unit: string, conditions?}]\n"
    "- conditions: string (pH, temperature, fermentation mode, etc.)\n"
    "- key_findings: [string] (1-5 bullets)\n"
    "- evidence: [{quote, where?}] (1-3 short quotes)\n"
    "- reaction_steps: [{substrate, product, enzyme? {name, ec_number?, organism?, reaction_type?, enzyme_class?, engineered?, mutations?[]}, notes?}]\n"
    "- strain_design: [{gene, action, details?}]\n\n"
    "Also provide narrative sections (concise paragraphs):\n"
    "- summary_long: string (what was done and why)\n"
    "- methods: string (key methodology and experimental setup)\n"
    "- results: string (key findings with numbers)\n"
    "- future_perspectives: string (next steps, limitations, outlook)\n\n"
    "Guidelines:\n"
    "- Ignore boilerplate (headers, nav, cookie notices).\n"
    "- Prefer specific numeric metrics (e.g., 95% yield, 275 g/L titer, 4 g/L/h productivity).\n"
    "- Include EC numbers if present.\n"
    "- If a field is unknown, omit it.\n"
)


def build_prompt(url: str, title: str | None, text: str) -> str:
    head = f"URL: {url}\nTITLE: {title or ''}\n\n"
    body = text
    # Keep a cap on input size for local models; caller should pre-truncate if necessary
    return PROMPT_TEMPLATE + "\n" + head + body
