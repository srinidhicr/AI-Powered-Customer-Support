# src/agents/orchestrator.py

import os, sys, json
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

from configs.config import config
from src.retrieval.semantic_cache import lookup, store

from src.tools.classify import classify
from src.tools.retrieve import retrieve
from src.tools.generate import generate
from src.tools.critique import critique
from src.tools.clarify  import clarify

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

1. ALWAYS call classify(query) first.
   If the message starts with [KNOWN CATEGORY: <cat>], extract the category,
   set confidence = 1.0, and SKIP calling classify again.

2. If confidence is extremely low or the query is clearly out of scope:
   → Call clarify(reason="out_of_scope") and STOP.

3. Confidence handling:
   - confidence < 0.25  → call retrieve(); if 0 docs → clarify(reason="low_confidence")
   - 0.25–0.65          → call retrieve(); continue with generate(); use critique() carefully
   - >= 0.65            → call retrieve() normally

4. After retrieve():
   → If no documents found → call clarify(reason="ambiguous") and STOP.
   → Otherwise call generate()

5. After generate():
   → Call critique()

6. If critique.should_retry == true AND retry_count < {config.max_retries}:
   → Retry retrieve() → generate() → critique()

7. Final output:
   → Return ONLY the final draft response as plain text.
   → Do NOT wrap it in JSON. Do NOT add commentary. Just the response text.

========================
RULES
========================
- Never hallucinate facts not in the retrieved documents.
- Prefer answering over unnecessary clarification.
- The final message in the conversation must be the plain-text draft for the agent.
"""

_agent = create_react_agent(
    model=_llm,
    tools=[classify, retrieve, generate, critique, clarify],
    prompt=SystemMessage(content=SYSTEM_PROMPT)
)


def get_final_draft(result: dict) -> str:
    """Extract the last meaningful plain-text AIMessage from the agent result."""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.strip() and not isinstance(msg, AIMessage):
            return content.strip()
        if not isinstance(msg, AIMessage):
            continue
        # Skip messages that are just tool-call invocations (no text content)
        if isinstance(content, list):
            texts = [i.get("text", "") for i in content if isinstance(i, dict)]
            text = " ".join(t for t in texts if t).strip()
            if text:
                return text
        if isinstance(content, str) and content.strip():
            # Skip if it looks like a tool-call thought with no actual answer
            c = content.strip()
            if len(c) > 30:   # ignore very short intermediary thoughts
                return c
    return ""


def extract_chunks_from_result(result: dict) -> list:
    """
    Walk the message list and find the ToolMessage that came from retrieve().
    Parse the documents out of it so the UI can display the source chunks.
    """
    messages = result.get("messages", [])
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        try:
            payload = json.loads(msg.content)
            docs = payload.get("documents", [])
            if docs:
                # Return a clean, serialisable summary of each chunk
                clean = []
                for d in docs[:5]:
                    clean.append({
                        "subject" : d.get("subject", ""),
                        "category": d.get("category", ""),
                        "answer"  : d.get("answer", d.get("text", ""))[:400],
                        "score"   : round(float(d.get("score", 0)), 4),
                        "rerank_score": round(float(d.get("rerank_score", 0)), 4),
                        "tags"    : d.get("tags", []),
                    })
                return clean
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
    return []


def answer_followup( previous_question: str, previous_response: str,
    followup_message: str, source_chunks: list | None = None) -> str:
    """
    Answer a clarification follow-up without re-running retrieval.
    This keeps the reply anchored to the previous answer and source context.
    """
    chunk_text = ""
    if source_chunks:
        chunk_lines = [
            f"- {chunk.get('answer', chunk.get('text', ''))[:300]}"
            for chunk in source_chunks[:3]
        ]
        chunk_text = "\n\nSupporting source context:\n" + "\n".join(chunk_lines)

    prompt = f"""
        You are helping continue an existing customer support conversation.
        Original customer question:
        {previous_question}
        f"Previous assistant response:
        {previous_response}
        Customer follow-up:
        {followup_message}
        {chunk_text}

        Instructions:
        1. Answer the follow-up as a clarification of the same issue.
        2. Stay on the same topic as the original question.
        3. Use simpler wording when the customer sounds confused.
        4. Do not ask a new routing question unless the previous answer truly lacked enough information.
        5. Do not introduce a different product, business scenario, or issue type.
    """

    response = _llm.invoke([
        SystemMessage(
            content=(
                "You are a customer support copilot. "
                "For clarification follow-ups, explain the last answer clearly and stay grounded in the provided context."
            )
        ),
        HumanMessage(content=prompt),
    ])
    content = getattr(response, "content", "")
    return content.strip() if isinstance(content, str) else ""


def run(query: str, use_cache: bool = True, forced_category: str = None) -> dict:
    """
    Run the full support pipeline for one customer message.

    Returns a dict with:
        messages      – raw LangGraph message list
        cache_hit     – bool
        cached_result – dict (only when cache_hit is True)
        chunks        – list of source documents used (for UI display)
    """
    # 1. Semantic cache check
    if use_cache:
        cached = lookup(query)
        if cached:
            return {
                "messages"     : [type("Msg", (), {"content": cached["final_draft"]})()],
                "cache_hit"    : True,
                "cached_result": cached,
                "chunks"       : cached.get("chunks", []),
            }

    # 2. Inject known category so classify() is skipped
    agent_query = query
    if forced_category:
        agent_query = f"[KNOWN CATEGORY: {forced_category}] {query}"

    result = _agent.invoke({"messages": [{"role": "user", "content": agent_query}]})

    final_draft = get_final_draft(result)
    chunks      = extract_chunks_from_result(result)

    # 3. Cache only substantive responses
    fallback_phrases = [
        "unable to find relevant information",
        "escalate this to a human agent",
        "please escalate"
    ]
    if (use_cache
            and final_draft
            and len(final_draft.split()) > 20
            and not any(p in final_draft.lower() for p in fallback_phrases)):
        store(query, {"final_draft": final_draft, "query": query, "chunks": chunks})

    result["cache_hit"] = False
    result["chunks"]    = chunks
    return result
