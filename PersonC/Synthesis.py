"""
Person C: LLM Synthesis & Guardrails
Generates root cause analysis from normalized evidence using Groq LLM API.
"""

import os
from typing import List
from groq import Groq
import json
import sys
from fastapi import APIRouter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.schemas import SynthesisRequest, SynthesisOut, Evidence, Driver


class SynthesisEngine:
    """Handles LLM-based synthesis of evidence into root cause analysis."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"
        self.max_tokens = 1024
        self.temperature = 0.3
    
    def _build_prompt(self, question: str, evidence: List[Evidence]) -> str:
        """Constructs guardrailed prompt for root cause analysis."""
        
        evidence_text = []
        for ev in evidence:
            dimension_info = f"\n[Evidence ID: {ev.id}] Dimension: {ev.dimension.upper()}"
            if ev.period:
                dimension_info += f" | Period: {ev.period}"
            
            kpi_text = []
            for key, value in ev.kpis.items():
                kpi_text.append(f"  - {key}: {value}")
            
            highlights_text = "\n".join([f"  • {h}" for h in ev.highlights])
            
            evidence_block = f"{dimension_info}\n"
            if kpi_text:
                evidence_block += "Metrics:\n" + "\n".join(kpi_text) + "\n"
            if highlights_text:
                evidence_block += f"Key Findings:\n{highlights_text}"
            
            evidence_text.append(evidence_block)
        
        all_evidence = "\n\n".join(evidence_text)
        
        prompt = f"""You are a data analyst assistant for Honeywell's Enterprise Data Warehouse. Analyze the provided evidence and explain WHY the situation occurred.

**USER QUESTION:**
{question}

**AVAILABLE EVIDENCE:**
{all_evidence}

**INSTRUCTIONS:**
1. Analyze all provided evidence to identify root causes
2. Explain WHY the situation occurred, not just what happened
3. Identify specific drivers and explain causal relationships
4. Cite evidence IDs that support each driver
5. Look for correlations across dimensions (time, region, product, orders)
6. Assess confidence: High (multiple consistent sources), Medium (limited evidence), Low (insufficient evidence)
7. State limitations or assumptions in the analysis
8. Suggest next steps for validation or deeper investigation

**OUTPUT FORMAT (JSON):**
{{
  "answer": "3-4 sentence explanation of WHY this happened, identifying root causes and their impact",
  "drivers": [
    {{
      "factor": "Specific root cause with causal explanation",
      "evidence_ids": ["e1", "e2"]
    }}
  ],
  "confidence": "High/Medium/Low",
  "limitations": ["Data gaps or assumptions in root cause analysis"],
  "next_steps": ["Suggested analyses to validate root causes"]
}}

**GUARDRAILS:**
- Only use information from provided evidence
- Do not fabricate numbers or facts
- Focus on explaining WHY, not just describing WHAT
- Identify causal relationships between dimensions
- Cite evidence IDs for transparency
- Keep explanation clear and business-focused

Generate the JSON response:"""
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Makes API call to Groq LLM."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data analyst specializing in root cause analysis. Explain WHY things happened by identifying causal relationships. Respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            raise Exception(f"Groq API call failed: {str(e)}")
    
    def _validate_and_repair(self, llm_output: str, evidence: List[Evidence]) -> SynthesisOut:
        """Validates LLM output against schema and repairs if needed."""
        try:
            parsed = json.loads(llm_output)
            valid_ids = {ev.id for ev in evidence}
            
            drivers = []
            for d in parsed.get("drivers", []):
                valid_evidence_ids = [eid for eid in d.get("evidence_ids", []) if eid in valid_ids]
                drivers.append(Driver(
                    factor=d.get("factor", "Unknown factor"),
                    evidence_ids=valid_evidence_ids
                ))
            
            confidence = parsed.get("confidence", "Medium")
            if confidence not in ["High", "Medium", "Low"]:
                confidence = "Medium"
            
            return SynthesisOut(
                answer=parsed.get("answer", "Unable to generate answer from available evidence."),
                drivers=drivers,
                confidence=confidence,
                limitations=parsed.get("limitations", ["Limited evidence available"]),
                next_steps=parsed.get("next_steps", ["Collect more detailed data"])
            )
        
        except json.JSONDecodeError as e:
            return SynthesisOut(
                answer="Error: Unable to parse LLM response.",
                drivers=[],
                confidence="Low",
                limitations=[f"JSON parsing error: {str(e)}"],
                next_steps=["Retry the query"]
            )
        except Exception as e:
            return SynthesisOut(
                answer="Error: Unable to generate synthesis.",
                drivers=[],
                confidence="Low",
                limitations=[f"Validation error: {str(e)}"],
                next_steps=["Check evidence format and retry"]
            )
    
    def synthesize(self, request: SynthesisRequest) -> SynthesisOut:
        """Main synthesis function."""
        prompt = self._build_prompt(request.question, request.evidence)
        llm_response = self._call_llm(prompt)
        synthesis_output = self._validate_and_repair(llm_response, request.evidence)
        return synthesis_output


_engine = None

def get_synthesis_engine() -> SynthesisEngine:
    """Returns singleton SynthesisEngine instance."""
    global _engine
    if _engine is None:
        _engine = SynthesisEngine()
    return _engine


def synthesize_answer(request: SynthesisRequest) -> SynthesisOut:
    """
    Synthesizes answer from evidence using LLM.
    Called by FastAPI endpoint.
    """
    engine = get_synthesis_engine()
    return engine.synthesize(request)


# FastAPI Router for Person A to import
router = APIRouter(prefix="/synth", tags=["synthesis"])

@router.post("/stub", response_model=SynthesisOut)
def synthesize_endpoint(req: SynthesisRequest) -> SynthesisOut:
    """Endpoint for synthesizing answers from evidence."""
    return synthesize_answer(req)
