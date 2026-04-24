# src/tools/generate.py
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
    max_tokens=500,
    temperature=0.3
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

class GenerateInput(BaseModel):
    query: str = Field(description="The original customer query.")
    category: str = Field(description="The predicted support category.")
    documents_json: Any = Field(
        description="JSON string or list of retrieved documents from retrieve() tool."
    )

    @field_validator("documents_json", mode="before")
    @classmethod
    def normalise_documents(cls, v) -> list:
        """
        Accept whatever the agent passes and always return a plain list of dicts.

        The LangGraph ReAct agent may pass documents_json as:
          - str  '{"documents": [...]}'   ← expected
          - str  '[{...}]'                ← direct list serialised
          - list [{...}]                  ← already deserialised
          - dict {"documents": [...]}     ← dict passed directly
        """
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

@tool(args_schema=GenerateInput)
def generate(query: str, category: str, documents_json: Any) -> str:
    """Generates a draft customer support response using retrieved documents.
    ALWAYS call retrieve() before this tool to get documents.
    Pass the full output of retrieve() as documents_json.
    Returns a JSON string with keys: draft_response, confidence, sources_used, caveats.
    """
    # After validation, documents_json is already a clean list
    documents: list = documents_json

    if not documents:
        return json.dumps({
            "draft_response": (
                "I was unable to find relevant information. "
                "Please escalate to a human agent."
            ),
            "confidence"  : 0.0,
            "sources_used": [],
            "caveats"     : "No documents retrieved.",
        })

    docs_text = "\n\n".join(
        f"[Doc {i+1}]: {d.get('answer', d.get('text', ''))}"
        for i, d in enumerate(documents[:5])
    )

    prompt = f"""You are a customer support copilot helping a support agent draft a response.

Support Category: {category}
Customer Query: {query}

Retrieved knowledge base context:
{docs_text}

Instructions:
1. Extract every concrete fact from the context above (system names, error types, known issues, policies, timelines).
2. Acknowledge the customer's specific problem using those facts.
3. If the context contains actionable steps, include them exactly.
4. If the context is acknowledgement-only, suggest 2-3 safe, generic next steps that are logically consistent with the issue type — do NOT invent specific technical details not implied by the context.
5. End with the appropriate escalation path if the issue cannot be resolved.

The response must stay grounded in the retrieved context. Do not introduce facts, product names, or solutions that are not implied by the context.

Return ONLY valid JSON with no markdown:
{{
  "draft_response": "full response text",
  "confidence": 0.85,
  "sources_used": ["Doc 1"],
  "caveats": "note any assumptions made"
}}"""

    raw   = _llm_invoke_with_retry(prompt)
    clean = re.sub(r'```json|```', '', raw).strip()

    if clean.startswith('{'):
        return clean

    return json.dumps({
        "draft_response": clean,
        "confidence"    : 0.5,
        "sources_used"  : [],
        "caveats"       : "Review carefully before sending.",
    })