# src/agents/orchestrator.py

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from configs.config import config
from src.retrieval.semantic_cache import lookup, store
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from src.tools.classify import classify
from src.tools.retrieve import retrieve
from src.tools.generate import generate
from src.tools.critique import critique
from src.tools.clarify  import clarify

"""
_llm = ChatGoogleGenerativeAI(
    model=config.llm_model,
    google_api_key=config.google_api_key
)
"""

_llm = ChatOpenAI(
    model=config.llm_model,
    api_key=config.openai_api_key,
    max_tokens=600,
    temperature=0
)

SYSTEM_PROMPT = f"""You are an autonomous orchestrator for a customer support copilot system.

Your goal is to help support agents by:
- Understanding customer queries
- Retrieving relevant knowledge
- Generating high-quality draft responses
- Iteratively improving responses when needed

You have access to the following tools:

- classify(query):
  Returns: category, confidence (0–1), urgency, in_scope

- retrieve(query, category):
  Returns: JSON with relevant documents

- generate(query, category, documents_json):
  Returns: JSON with draft_response, confidence, sources_used, caveats

- critique(query, draft_json, documents_json):
  Returns: JSON with evaluation scores and should_retry flag

- clarify(query, reason):
  Returns: JSON with a clarifying question for the user


========================
DECISION POLICY (STRICT)
========================

1. ALWAYS call classify(query) first.

2. If in_scope is False:
   → Call clarify(reason="out_of_scope") and STOP.

3. Confidence handling (IMPORTANT — DO NOT IGNORE):

   - If confidence < 0.45:
       → Call clarify(reason="low_confidence") and STOP.

   - If 0.45 ≤ confidence < 0.65:
       → Proceed with retrieve(query, category)
       → Treat this as LOW confidence internally
       → Rely on critique() to validate and improve

   - If confidence ≥ 0.65:
       → Proceed normally with retrieve(query, category)

4. After retrieve():
   → Call generate(query, category, documents_json)

5. After generate():
   → Call critique(query, draft_json, documents_json)

6. If critique.should_retry == true AND retry_count < {config.max_retries}:
   → Call retrieve() again, refining the query using critique feedback
   → Call generate() again
   → Call critique() again

7. Stop when:
   - critique.should_retry == false
   OR
   - max retries reached

8. FINAL OUTPUT:
   → Return ONLY the final draft response (plain text)
   → DO NOT return tool traces, JSON, or intermediate steps


========================
IMPORTANT RULES
========================

- DO NOT stop early just because confidence is moderate (0.45–0.65)
- The system is designed to recover using retrieval + critique
- Prefer acting over asking, unless confidence is VERY low

- ALWAYS pass FULL JSON outputs between tools
- NEVER summarize or truncate tool outputs

- DO NOT hallucinate facts — rely on retrieved documents
- If no useful documents are found, generate a safe fallback response

- The final answer MUST be a clean, human-readable string
- DO NOT return lists, objects, or JSON in the final output


========================
BEHAVIOR SUMMARY
========================

- Be autonomous, not overly cautious
- Use tools intelligently
- Iterate when needed
- Ask clarification ONLY when truly necessary
"""

_agent = create_react_agent(
    model=_llm,
    tools=[classify, retrieve, generate, critique, clarify],
    prompt=SystemMessage(content=SYSTEM_PROMPT)
)

def get_final_draft(result: dict) -> str:
    messages = result.get("messages", [])
    if not messages:
        return ""

    from langchain_core.messages import AIMessage
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        content = msg.content
        
        if isinstance(content, str) and content.strip():
            return content.strip()
        
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    # print(f"[DEBUG dict keys]: {list(item.keys())}") 
                    for key in ("text", "content", "parts", "output", "response"):
                        val = item.get(key)
                        if isinstance(val, str) and val.strip():
                            return val.strip()
                        if isinstance(val, list):
                            for part in val:
                                if isinstance(part, dict) and part.get("text"):
                                    return part["text"].strip()
                                if isinstance(part, str) and part.strip():
                                    return part.strip()
                elif isinstance(item, str) and item.strip():
                    return item.strip()
    return ""


def run(query: str, use_cache: bool = True) -> dict:
    """Run the agent on a customer query, with optional semantic cache."""

    # 1. Check cache first
    if use_cache:
        cached = lookup(query)
        if cached:
            return {
                "messages": [
                    type("Msg", (), {"content": cached["final_draft"]})()
                ],
                "cache_hit"    : True,
                "cached_result": cached
            }

    # 2. Run the full agent pipeline
    result      = _agent.invoke({"messages": [{"role": "user", "content": query}]})
    final_draft = get_final_draft(result)

    # 3. Store in cache only if response is substantive (not a fallback)
    fallback_phrases = [
        "unable to find relevant information",
        "escalate this to a human agent",
        "please escalate"
    ]
    if (use_cache
            and final_draft
            and len(final_draft.split()) > 20
            and not any(p in final_draft.lower() for p in fallback_phrases)):
        store(query, {"final_draft": final_draft, "query": query})

    result["cache_hit"] = False
    return result