# src/tools/critique.py

import os
import sys
import json
import re
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from configs.config import config
import time

"""
_llm = ChatGoogleGenerativeAI(
    model=config.llm_model,
    google_api_key=config.google_api_key
)

def _llm_invoke_with_retry(prompt: str, max_retries: int = 1) -> str:
    for attempt in range(max_retries):
        try:
            return _llm.invoke(prompt).content
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 45 * (attempt + 1)
                print(f"[Rate limit] waiting {wait}s before retry {attempt+1}/{max_retries}")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded")
"""
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


import time

def _llm_invoke_with_retry(prompt: str, max_retries: int = 1) -> str:
    for attempt in range(max_retries):
        try:
            return _llm.invoke(prompt).content
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 45 * (attempt + 1)
                print(f"[Rate limit] waiting {wait}s before retry {attempt+1}/{max_retries}")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded")

@tool
def critique(query: str, draft_json: str, documents_json: str) -> str:
    """Evaluates a draft response for quality, grounding and tone.
    ALWAYS call generate() before this tool.
    If overall_score < 0.70, the orchestrator should retry retrieve() and generate().

    Args:
        query:          The original customer query.
        draft_json:     JSON string output from generate() tool.
        documents_json: JSON string output from retrieve() tool.

    Returns:
        JSON string with keys: grounding_score, tone_score, completeness_score,
        overall_score, hallucination_flag, feedback, should_retry
    """
    # Parse draft
    try:
        draft_result = json.loads(draft_json)
        draft = draft_result.get("draft_response", draft_json)
    except (json.JSONDecodeError, TypeError):
        draft = str(draft_json)

    # Parse documents
    try:
        retrieve_result = json.loads(documents_json)
        documents = retrieve_result.get("documents", [])
    except (json.JSONDecodeError, TypeError):
        documents = []

    docs_text = "\n".join(
        f"- {d.get('text', d.get('body', ''))[:300]}"
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
            "feedback"           : "Could not parse evaluation — review manually."
        }

    result["should_retry"] = result.get("overall_score", 0) < config.critique_threshold
    return json.dumps(result)