from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    quote: str = Field(..., description="Direct quote or minimal span supporting a fact")
    where: Optional[str] = Field(None, description="Section/heading if obvious")


class Metric(BaseModel):
    kind: str = Field(..., description="metric type: yield|titer|productivity")
    value: float
    unit: str
    conditions: Optional[str] = None


class Enzyme(BaseModel):
    name: str
    ec_number: Optional[str] = None
    organism: Optional[str] = None
    reaction_type: Optional[str] = None
    enzyme_class: Optional[str] = None
    engineered: Optional[bool] = None
    mutations: Optional[list[str]] = None


class Organism(BaseModel):
    name: str
    role: Optional[str] = None


class PageSummary(BaseModel):
    url: str
    title: Optional[str]
    year: Optional[int] = None
    journal_or_source: Optional[str] = None
    chemical: Optional[str] = None
    approach: Optional[str] = Field(None, description="e.g., microbial, cell-free, engineered pathway, fermentation")
    pathway: Optional[str] = None
    organisms: List[Organism] = []
    enzymes: List[Enzyme] = []
    feedstocks: List[str] = []
    starting_substrates: List[str] = []
    metrics: List[Metric] = []
    conditions: Optional[str] = None
    key_findings: List[str] = []
    evidence: List[Evidence] = []
    reaction_steps: List["ReactionStep"] = []
    strain_design: List["StrainModification"] = []
    # Narrative sections
    summary_long: Optional[str] = None
    methods: Optional[str] = None
    results: Optional[str] = None
    future_perspectives: Optional[str] = None

class ReactionStep(BaseModel):
    substrate: str
    product: str
    enzyme: Optional[Enzyme] = None
    notes: Optional[str] = None


class StrainModification(BaseModel):
    gene: str
    action: str = Field(..., description="e.g., deletion|knockout|overexpression|upregulation|downregulation")
    details: Optional[str] = None


PageSummary.model_rebuild()
