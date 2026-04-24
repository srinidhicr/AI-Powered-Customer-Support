# src/tools/critique.py
import os, sys, json, re, time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator
from typing import Any
from configs.config import config

_llm = ChatOpenAI(
    model=config.llm_model,
    api_key=config.openai_api_key,
    max_tokens=300,
    temperature=0
)

def _llm_invoke_with_retry(prompt: str) -> str:
    for attempt in range(3):
        try:
            return _llm.invoke(prompt).content.strip()
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = 30 * (attempt + 1)
                print(f"[Rate limit] waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded")


# ── Pydantic input schema ─────────────────────────────────────────────────────

class CritiqueInput(BaseModel):
    query: str = Field(description="The original customer query.")
    draft_json: Any = Field(
        description="JSON string or dict output from generate() tool."
    )
    documents_json: Any = Field(
        description="JSON string or list of retrieved documents from retrieve() tool."
    )

    @field_validator("draft_json", mode="before")
    @classmethod
    def normalise_draft(cls, v) -> str:
        """Always return the plain draft text string."""
        if isinstance(v, dict):
            return v.get("draft_response", str(v))
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, dict):
                    return parsed.get("draft_response", v)
            except (json.JSONDecodeError, ValueError):
                pass
            return v
        return str(v)

    @field_validator("documents_json", mode="before")
    @classmethod
    def normalise_documents(cls, v) -> list:
        """Always return a plain list of document dicts."""
        if isinstance(v, list):
            return v
        if isinstance(v, dict):
            return v.get("documents", [])
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
                if isinstance(parsed, dict):
                    return parsed.get("documents", [])
            except (json.JSONDecodeError, ValueError):
                pass
        return []


# ── Tool ──────────────────────────────────────────────────────────────────────

@tool(args_schema=CritiqueInput)
def critique(query: str, draft_json: Any, documents_json: Any) -> str:
    """Evaluates a draft response for quality, grounding and tone.
    ALWAYS call generate() before this tool.
    If overall_score < 0.70, the orchestrator should retry retrieve() and generate().
    Returns a JSON string with keys: grounding_score, tone_score, completeness_score,
    overall_score, hallucination_flag, feedback, should_retry.
    """
    # After Pydantic validation these are already clean types
    draft: str    = draft_json
    documents: list = documents_json

    docs_text = "\n".join(
        f"- {d.get('text', d.get('answer', d.get('body', '')))[:300]}"
        for d in documents[:5]
    ) or "No source documents available."

    prompt = f"""You are a quality evaluator for customer support responses.

Customer Query: {query}

Draft Response:
{draft}

Source Documents Used:
{docs_text}

Evaluate the draft response on these dimensions. Score each 0.0 to 1.0:
- grounding_score: Is every claim in the draft supported by the source documents?
- tone_score: Is the tone professional, empathetic and appropriate?
- completeness_score: Does the response fully address the customer query?
- overall_score: Weighted average of the above scores.
- hallucination_flag: true if the draft contains claims NOT in the source documents.
- feedback: Specific actionable feedback if overall_score < 0.70, else empty string.

Return ONLY valid JSON with no markdown:
{{
  "grounding_score": 0.0,
  "tone_score": 0.0,
  "completeness_score": 0.0,
  "overall_score": 0.0,
  "hallucination_flag": false,
  "feedback": ""
}}"""

    raw   = _llm_invoke_with_retry(prompt)
    clean = re.sub(r'```json|```', '', raw).strip()

    try:
        result = json.loads(clean)
    except json.JSONDecodeError:
        result = {
            "grounding_score"    : 0.5,
            "tone_score"         : 0.5,
            "completeness_score" : 0.5,
            "overall_score"      : 0.5,
            "hallucination_flag" : False,
            "feedback"           : "Could not parse evaluation — review manually.",
        }

    result["should_retry"] = result.get("overall_score", 0) < config.critique_threshold
    return json.dumps(result)