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
    max_tokens=4000,
    temperature=0
)

SYSTEM_PROMPT = f"""
You are an autonomous orchestrator for a customer support copilot system.

========================
DECISION POLICY
========================

1. ALWAYS call classify(query) first. If a known category is provided in the user message, trust it and skip reclassification unless the user clearly changed topics.

2. If in_scope is False:
   → Call clarify(reason="out_of_scope") and STOP.

3. Confidence handling:

   - If confidence < 0.25:
       → This is very uncertain.
       → First call retrieve(query, category)

       If retrieved documents count == 0:
           → Call clarify(reason="low_confidence")
           → STOP

       Else:
           → Continue with generate()

   - If 0.25 <= confidence < 0.65:
       → Proceed with retrieve(query, category)
       → Use critique() carefully
       → Do NOT clarify immediately

   - If confidence >= 0.65:
       → Proceed normally with retrieve(query, category)

4. After retrieve():
   → If no documents found:
       → Call clarify(reason="ambiguous")

5. After retrieve():
   → Call generate()

6. After generate():
   → Call critique()

7. If critique.should_retry == true AND retry_count < {config.max_retries}:
   → Retry retrieve()
   → Retry generate()
   → Retry critique()

8. Final output:
   → Return ONLY final draft response text.

========================
IMPORTANT RULES
========================

- Retrieval can succeed even when classification confidence is low.
- Prefer answering over asking unnecessary clarification.
- Use classifier as guidance, not as a hard blocker.
- Never hallucinate facts.
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


def run(query: str, use_cache=True, forced_category=None):
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
    if forced_category:
        query = f"[KNOWN CATEGORY: {forced_category}] {query}"
    result      = _agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = result.get("messages", [])
    print(f"[DEBUG] total messages: {len(messages)}")
    for i, msg in enumerate(messages):
        content = getattr(msg, 'content', '')
        content_type = type(content).__name__
        preview = str(content)[:80] if isinstance(content, str) else str(content)[:80]
        print(f"[DEBUG] msg[{i}] type={type(msg).__name__} content_type={content_type} preview={preview}")
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