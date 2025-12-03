from dotenv import load_dotenv
load_dotenv()

import re, uuid, os, requests
from fastapi import FastAPI, APIRouter

from groq import Groq

# allow package-style imports from repo root
import sys, os as _os
sys.path.append(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.append(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from shared.schemas import (
    RouteRequest, RouteResponse, DecomposeResponse, SubQuestion,
    SynthesisRequest
)

# --- LLM planner config (optional, hybrid mode) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PLANNER_MODEL = "llama-3.3-70b-versatile"

planner_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Import B & C routers
from PersonB.FetchNormalize import router as data_router
from PersonC.Synthesis import router as synth_router

app = FastAPI(
    title="EDW Reasoning Assistant - Alpha",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- A: routing & decomposition ----------
route_router = APIRouter(prefix="/route", tags=["routing"])
REASONING_PAT = re.compile(r"\b(why|cause|reason|decline|drop|drivers?)\b", re.I)

@route_router.post("", response_model=RouteResponse)
def route(req: RouteRequest) -> RouteResponse:
    q = (req.question or "").strip()
    return RouteResponse(type="reasoning" if REASONING_PAT.search(q) else "basic")

def refine_subquestions_with_llm(
    question: str,
    defaults: list[tuple[str, str]],
) -> dict[str, str]:
    """
    Hybrid planner:
    - Takes the user's question + default (dimension, nlq) pairs.
    - Asks the LLM to rewrite / specialize the NL queries per dimension.
    - Returns a mapping {dimension: refined_nlq}.
    - On any error or missing API key, returns {} (use defaults).
    """
    if planner_client is None or not GROQ_API_KEY:
        # No key configured -> skip refinement, use defaults
        return {}

    # Build a simple JSON-oriented prompt
    default_blocks = "\n".join(
        [f"- dimension: {dim}, nlq: {nlq}" for dim, nlq in defaults]
    )

    prompt = f"""
You are a planning agent for a business data reasoning assistant.
The user has asked the following question:

USER QUESTION:
{question}

We currently decompose questions into four standard dimensions:
time, region, product, and orders.

For each dimension, you will produce ONE focused natural language query (nlq)
that will help answer the user's question, staying consistent with that dimension.

Defaults (for reference):
{default_blocks}

Your job:
- Refine or specialize these queries to match the user's question.
- Keep exactly these dimensions: time, region, product, orders.
- Do not invent new dimensions.
- If the user's question does not clearly relate to one dimension,
  still produce a reasonable generic query for that dimension.

Return ONLY valid JSON in this format:

{{
  "sub_questions": [
    {{"dimension": "time", "nlq": "..." }},
    {{"dimension": "region", "nlq": "..." }},
    {{"dimension": "product", "nlq": "..." }},
    {{"dimension": "orders", "nlq": "..." }}
  ]
}}
"""

    try:
        res = planner_client.chat.completions.create(
            model=PLANNER_MODEL,
            messages=[
                {"role": "system",
                 "content": "You are a careful planning agent. Respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=512,
        )
        content = res.choices[0].message.content
        data = json.loads(content)

        refined: dict[str, str] = {}
        for item in data.get("sub_questions", []):
            dim = (item.get("dimension") or "").strip().lower()
            nlq = (item.get("nlq") or "").strip()
            if dim and nlq:
                refined[dim] = nlq

        return refined
    except Exception as e:
        # In alpha we can just log and fall back
        print(f"[planner] LLM refinement failed: {e}")
        return {}
    
@route_router.post("/decompose", response_model=DecomposeResponse)
def decompose(req: RouteRequest) -> DecomposeResponse:
    """
    Hybrid decomposition:
    - Start with hardcoded defaults (time/region/product/orders).
    - Optionally refine the `nlq` text with an LLM, based on the user's question.
    - Dimensions and IDs remain stable; only the nlq text may change.
    """
    sid = lambda: uuid.uuid4().hex[:6]

    # 1) Hardcoded defaults (safe baseline)
    default_pairs: list[tuple[str, str]] = [
        ("time",    "Analyze revenue QoQ across last two quarters."),
        ("region",  "Compare revenue by region across the last quarter."),
        ("product", "Compare revenue by product line for the last quarter."),
        ("orders",  "Analyze order volume and average order value QoQ."),
    ]

    # 2) Ask LLM to refine them (if configured)
    refined_map = refine_subquestions_with_llm(req.question or "", default_pairs)

    # 3) Build final SubQuestion objects
    subqs: list[SubQuestion] = []
    for dim, default_nlq in default_pairs:
        nlq = refined_map.get(dim, default_nlq)  # fall back to default if no refinement
        subqs.append(
            SubQuestion(
                id=f"q_{dim}_{sid()}",
                dimension=dim,
                nlq=nlq,
            )
        )

    return DecomposeResponse(sub_questions=subqs)

app.include_router(route_router)   # A
app.include_router(data_router)    # B
app.include_router(synth_router)   # C

@app.get("/")
def root():
    return {"ok": True, "service": "edw-alpha"}

# ---------- Optional: one-call orchestrator ----------
ask_router = APIRouter(prefix="/ask", tags=["orchestrator"])
BASE = os.getenv("SELF_BASE", "http://localhost:8000")

@ask_router.post("")
def ask(req: RouteRequest):
    rtype = requests.post(f"{BASE}/route", json=req.dict()).json()["type"]
    if rtype != "reasoning":
        return {"answer": "Basic path not implemented in alpha."}
    subqs = requests.post(f"{BASE}/route/decompose", json=req.dict()).json()["sub_questions"]
    results = requests.post(f"{BASE}/data/fetch", json={"sub_questions": subqs}).json()["results"]
    evidence = requests.post(f"{BASE}/data/normalize", json={"results": results}).json()["evidence"]
    out = requests.post(f"{BASE}/synth/stub",
                        json=SynthesisRequest(question=req.question, evidence=evidence).dict()).json()
    return {"evidence": evidence, "synthesis": out}

app.include_router(ask_router)

