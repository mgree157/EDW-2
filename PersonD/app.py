import os
import json
import time
from typing import Any, Dict, List

import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ==========================================================
# CONFIGURATION
# ==========================================================
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")  # FastAPI backend base URL
ROUTE_EP = f"{BACKEND_URL}/route"
ANSWER_EP = f"{BACKEND_URL}/ask"
HEALTH_EP = f"{BACKEND_URL}/health"

st.set_page_config(page_title="EDW Reasoning Assistant", page_icon="🧠", layout="wide")

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def ping_health() -> Dict[str, Any]:
    """Check the /health endpoint and measure latency."""
    try:
        t0 = time.time()
        r = requests.get(HEALTH_EP, timeout=5)
        latency = round((time.time() - t0) * 1000, 1)
        if r.ok:
            data = r.json()
            data["ok"] = True
            data["latency_ms"] = latency
            return data
        return {"ok": False, "error": f"{r.status_code}: {r.text}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Send POST request to backend with timing and error handling."""
    try:
        t0 = time.time()
        r = requests.post(url, json=payload, timeout=60)
        elapsed = round(time.time() - t0, 2)
        if r.ok:
            data = r.json()
            data["_request_time_s"] = elapsed
            return data
        return {"_error": f"{r.status_code} {r.text}", "_request_time_s": elapsed}
    except Exception as e:
        return {"_error": str(e), "_request_time_s": 0.0}


def safe_get(d: Dict, key: str, default=None):
    """Avoid KeyErrors on missing JSON fields."""
    return d.get(key, default)


def show_json_copy_download(label: str, data: Dict[str, Any]):
    """Show pretty JSON, and let user copy or download it."""
    pretty = json.dumps(data, indent=2)
    st.subheader(label)
    st.code(pretty, language="json")

    st.download_button(
        "⬇️ Download JSON trace",
        data=pretty.encode("utf-8"),
        file_name="trace.json",
        mime="application/json",
        use_container_width=True
    )

    # Simple Copy to Clipboard using HTML
    components.html(
        f"""
        <textarea id="jsontrace" rows="10" style="width:100%;border:1px solid #444;
            border-radius:8px;padding:8px;">{pretty}</textarea>
        <button onclick="navigator.clipboard.writeText(
            document.getElementById('jsontrace').value)" 
            style="margin-top:8px;padding:8px 12px;border-radius:8px;border:0;cursor:pointer;">
            📋 Copy JSON to clipboard
        </button>
        """,
        height=260,
    )


# ==========================================================
# SIDEBAR SETTINGS
# ==========================================================
st.sidebar.title("⚙️ Settings")

with st.sidebar:
    st.caption("Backend Health")
    health = ping_health()
    if health.get("ok"):
        st.success(f"Healthy • {health['latency_ms']} ms")
    else:
        st.error(f"Unhealthy: {health.get('error', 'unknown')}")

    st.caption("Endpoints")
    st.write(f"- `/route`: {ROUTE_EP}")
    st.write(f"- `/answer`: {ANSWER_EP}")
    st.write(f"- `/health`: {HEALTH_EP}")

    # Control toggles
    auto_detect = st.toggle("Auto-detect reasoning queries (/route)", value=True)
    show_debug = st.toggle("Show backend payloads", value=False)

    # NEW: Demo mode toggle
    mode = st.radio("Mode", ["API", "Demo (local JSON)"], horizontal=True)
    demo_path = st.text_input("Demo JSON path", value="person_d/demo_response.json")

# ==========================================================
# MAIN INTERFACE
# ==========================================================
st.title("🧠 EDW Reasoning Assistant")
st.write(
    "Ask a question about business data — the system will route it through reasoning, "
    "data fetching, normalization, and synthesis."
)

# User input form
with st.form("ask"):
    user_q = st.text_input("Your Question:", placeholder="e.g., Why did Q2 revenue drop compared to Q1?")
    submitted = st.form_submit_button("Run Reasoning", use_container_width=True)

# ==========================================================
# RUN PIPELINE (BACKEND OR DEMO)
# ==========================================================
if submitted and user_q.strip():
    st.divider()
    colA, colB = st.columns([2, 1])

    # ----------------------------
    # 1. Route or Demo
    # ----------------------------
    if mode == "API":
        route_res = post_json(ROUTE_EP, {"question": user_q}) if auto_detect else {}
        if show_debug and auto_detect:
            with st.expander("Raw /route response"):
                st.json(route_res)

        answer_res = post_json(ANSWER_EP, {"question": user_q})
        # Adapt /ask response (evidence + synthesis) to the UI schema
        if "evidence" in answer_res and "synthesis" in answer_res:
            syn = answer_res.get("synthesis", {})
            evidence = answer_res.get("evidence", [])
            answer_res = {
                "final_answer": syn.get("answer", "No final answer provided."),
                # Backend uses string confidence ("High"/"Medium"/"Low"), but the
                # UI expects a numeric 0–1; we leave it None so it shows "—"
                "confidence": None,
                "sub_questions": [],          
                "evidence": evidence,
                "findings": syn.get("drivers", []),
                "timings": {},                
                "trace": {
                    "route_type": route_res.get("type") if isinstance(route_res, dict) else None,
                    "raw": answer_res,
                },
            }
    else:
        # Demo mode loads static JSON
        try:
            with open(demo_path, "r", encoding="utf-8") as f:
                answer_res = json.load(f)
            answer_res["_request_time_s"] = 0.01
        except Exception as e:
            st.error(f"Failed to load demo JSON: {e}")
            st.stop()

    # ----------------------------
    # 2. Handle backend error
    # ----------------------------
    if "_error" in answer_res:
        st.error(f"Backend error: {answer_res['_error']}")
        st.stop()

    # ----------------------------
    # 3. Extract key fields
    # ----------------------------
    final_answer = safe_get(answer_res, "final_answer", "No final answer provided.")
    confidence = safe_get(answer_res, "confidence")
    sub_questions = safe_get(answer_res, "sub_questions", [])
    evidence = safe_get(answer_res, "evidence", [])
    findings = safe_get(answer_res, "findings", [])
    timings = safe_get(answer_res, "timings", {})
    trace = safe_get(answer_res, "trace", answer_res)

    # ----------------------------
    # 4. Display final answer
    # ----------------------------
    with colA:
        st.subheader("Final Answer")
        st.write(final_answer)

    with colB:
        st.metric("Confidence", f"{round(confidence*100,1)}%" if confidence else "—")
        req_time = answer_res.get("_request_time_s", None)
        if req_time:
            st.metric("Request Time", f"{req_time:.2f}s")

    # ----------------------------
    # 5. Sub-questions
    # ----------------------------
    with st.expander("🔎 Sub-questions"):
        if sub_questions:
            for i, q in enumerate(sub_questions, start=1):
                st.markdown(f"**{i}.** {q}")
        else:
            st.info("No sub-questions returned.")

    # ----------------------------
    # 6. Evidence
    # ----------------------------
    with st.expander("🧾 Evidence"):
        if evidence:
            try:
                df = pd.DataFrame(evidence)
                st.dataframe(df, use_container_width=True, hide_index=True)
            except Exception:
                st.json(evidence)
            st.caption("Each evidence item should include an `evidence_id` and source file (mock CSV).")
        else:
            st.info("No evidence returned.")

    # ----------------------------
    # 7. Findings
    # ----------------------------
    with st.expander("📌 Findings"):
        if findings:
            for f in findings:
                if isinstance(f, dict):
                    st.write(f"- **{f.get('title', 'Finding')}** — {f.get('summary', '')}")
                else:
                    st.write(f"- {str(f)}")
        else:
            st.info("No findings returned.")

    # ----------------------------
    # 8. Observability (Timings)
    # ----------------------------
    with st.expander("⏱️ Timings & Observability"):
        if timings:
            cols = st.columns(min(4, len(timings)))
            for i, (k, v) in enumerate(timings.items()):
                try:
                    cols[i % len(cols)].metric(k, f"{float(v):.2f}s")
                except Exception:
                    cols[i % len(cols)].metric(k, str(v))

            try:
                tdf = pd.DataFrame(
                    [{"stage": k, "seconds": float(v)} for k, v in timings.items()]
                ).set_index("stage")
                st.bar_chart(tdf["seconds"])
            except Exception:
                st.json(timings)
        else:
            st.info("No timing data available.")

    # ----------------------------
    # 9. JSON Trace (Download + Copy)
    # ----------------------------
    show_json_copy_download("🧬 JSON Trace", trace)

    if show_debug:
        with st.expander("Raw /answer Response"):
            st.json(answer_res)

else:
    st.info("Enter a question above and click **Run Reasoning** to begin.")
