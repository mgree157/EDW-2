from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# ---------- A: Routing ----------
class RouteRequest(BaseModel):
    question: str

class RouteResponse(BaseModel):
    type: str  # "basic" | "reasoning"

class SubQuestion(BaseModel):
    id: str
    dimension: str   # time | region | product | orders
    nlq: str

class DecomposeResponse(BaseModel):
    sub_questions: List[SubQuestion]

from typing import Literal
class PlanStep(BaseModel):
    id: str
    kind: Literal["query", "synthesis"]
    description: str
    dimension: Optional[str] = None   # "time", "region", "product", "orders", etc.
    nlq: Optional[str] = None         # natural language query for data steps
    depends_on: List[str] = []        # ids of steps this depends on

class Plan(BaseModel):
    question: str
    steps: List[PlanStep]
    
# ---------- B: Data ----------
class RawResult(BaseModel):
    id: str
    dimension: str
    rows: List[Dict[str, Any]] = Field(default_factory=list)

class FetchRequest(BaseModel):
    sub_questions: List[SubQuestion]

class FetchResponse(BaseModel):
    results: List[RawResult]

class Evidence(BaseModel):
    id: str
    dimension: str
    period: Optional[str] = None
    kpis: Dict[str, float] = Field(default_factory=dict)
    highlights: List[str] = Field(default_factory=list)

class NormalizeRequest(BaseModel):
    results: List[RawResult]

class NormalizeResponse(BaseModel):
    evidence: List[Evidence]

# ---------- C: Synthesis ----------
class Driver(BaseModel):
    factor: str
    evidence_ids: List[str] = Field(default_factory=list)

class SynthesisRequest(BaseModel):
    question: str
    evidence: List[Evidence]

class SynthesisOut(BaseModel):
    answer: str
    drivers: List[Driver] = Field(default_factory=list)
    confidence: str
    limitations: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
