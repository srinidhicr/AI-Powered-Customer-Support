# src/tools/generate.py
import os, sys, json, re, time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from langchain.tools import tool
from langchain_openai import ChatOpenAI
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

@tool
def generate(query: str, category: str, documents_json: str) -> str:
    """Generates a draft customer support response using retrieved documents.
    ALWAYS call retrieve() before this tool to get documents.

    Args:
        query:          The original customer query.
        category:       The predicted support category.
        documents_json: JSON string of retrieved documents from retrieve() tool.

    Returns:
        JSON string with keys: draft_response, confidence, sources_used, caveats
    """
    try:
        documents = json.loads(documents_json).get("documents", [])
    except (json.JSONDecodeError, TypeError):
        documents = []

    if not documents:
        return json.dumps({
            "draft_response": "I was unable to find relevant information. Please escalate to a human agent.",
            "confidence"    : 0.0,
            "sources_used"  : [],
            "caveats"       : "No documents retrieved."
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

    try:
        return clean if clean.startswith('{') else json.dumps({
            "draft_response": clean,
            "confidence"    : 0.5,
            "sources_used"  : [],
            "caveats"       : "Review carefully before sending."
        })
    except Exception:
        return json.dumps({
            "draft_response": clean,
            "confidence"    : 0.5,
            "sources_used"  : [],
            "caveats"       : "Could not parse structured output."
        })