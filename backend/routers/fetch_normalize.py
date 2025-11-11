from fastapi import APIRouter
from typing import List, Dict, Any
import math

from backend.schemas import (
    FetchRequest, FetchResponse, NormalizeRequest, NormalizeResponse,
    RawResult, Evidence
)
from backend.adapters.mock_cortex_adapter import MockCortexAdapter

router = APIRouter(prefix="/data", tags=["data"])
adapter = MockCortexAdapter()

def _safe_num(v):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)): return None
        return float(v)
    except Exception:
        return None

@router.post("/fetch", response_model=FetchResponse)
def fetch(req: FetchRequest) -> FetchResponse:
    results: List[RawResult] = []
    for sq in req.sub_questions:
        rows = adapter.run_nl(sq.nlq, sq.dimension)
        results.append(RawResult(id=sq.id, dimension=sq.dimension, rows=rows))
    return FetchResponse(results=results)

@router.post("/normalize", response_model=NormalizeResponse)
def normalize(req: NormalizeRequest) -> NormalizeResponse:
    evidence_list: List[Evidence] = []
    for r in req.results:
        dim = (r.dimension or "").lower()
        rows: List[Dict[str, Any]] = r.rows or []
        ev = Evidence(id=r.id, dimension=r.dimension, period=None, kpis={}, highlights=[])

        if dim == "time":
            if rows:
                latest = rows[-1]
                ev.period = str(latest.get("period"))
                ev.kpis["revenue"] = _safe_num(latest.get("revenue")) or 0.0
                d = _safe_num(latest.get("rev_delta_pct"))
                if d is not None:
                    ev.kpis["rev_delta_pct"] = d
                    ev.highlights.append(f"Revenue {d:+.0f}% QoQ")
                else:
                    ev.highlights.append("Revenue trend available (delta missing)")
            else:
                ev.highlights.append("No time-series data available")

        elif dim == "region":
            if rows:
                worst = rows[0]  # sorted ascending by rev_delta_pct
                ev.period = str(worst.get("period"))
                ev.kpis["revenue"] = _safe_num(worst.get("revenue")) or 0.0
                d = _safe_num(worst.get("rev_delta_pct")) or 0.0
                ev.kpis["rev_delta_pct"] = d
                ev.highlights.append(f"{worst.get('region','Unknown')} {d:+.0f}% QoQ")
            else:
                ev.highlights.append("No regional breakdown available")

        elif dim == "product":
            if rows:
                worst = rows[0]
                ev.period = str(worst.get("period"))
                ev.kpis["revenue"] = _safe_num(worst.get("revenue")) or 0.0
                d = _safe_num(worst.get("rev_delta_pct")) or 0.0
                ev.kpis["rev_delta_pct"] = d
                ev.highlights.append(f"{worst.get('product','Unknown')} {d:+.0f}% QoQ")
            else:
                ev.highlights.append("No product breakdown available")

        elif dim == "orders":
            if rows:
                latest = rows[-1]
                ev.period = str(latest.get("period"))
                for k in ("orders", "aov", "orders_delta_pct"):
                    v = _safe_num(latest.get(k))
                    if v is not None:
                        ev.kpis[k] = v
                if "orders_delta_pct" in ev.kpis:
                    ev.highlights.append(f"Orders {ev.kpis['orders_delta_pct']:+.0f}% QoQ")
                else:
                    ev.highlights.append("Orders trend available (delta missing)")
            else:
                ev.highlights.append("No orders data available")

        else:
            ev.highlights.append("Unknown dimension")

        evidence_list.append(ev)

    return NormalizeResponse(evidence=evidence_list)

