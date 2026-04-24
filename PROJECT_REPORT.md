# Project Report

Last updated: 2026-04-23

This is the living project reference for the customer-support chatbot. It is meant to be updated as we continue working, so the current architecture, major fixes, known issues, and next steps stay in one place.

## 1. Project Goal

Build a customer-support chatbot that:

- opens and tracks support tickets
- keeps chat continuity within a ticket
- routes unrelated questions away from the active ticket
- retrieves the most relevant knowledge-base chunks
- generates grounded support responses instead of generic or off-topic replies

## 2. High-Level Architecture

Current main flow:

1. UI receives a message in [src/ui/chat_handler.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/chat_handler.py)
2. Slash commands are handled in [src/ui/components/commands.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/components/commands.py)
3. Basic guardrails run in [src/ui/guardrails.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/guardrails.py)
4. Dedicated turn-intent routing runs in [src/ui/turn_intent.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/turn_intent.py)
5. Follow-up handling and conversation context are assembled in [src/ui/chat_handler.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/chat_handler.py) and [src/ui/ticket_helpers.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/ticket_helpers.py)
6. The orchestrator in [src/agents/orchestrator.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/agents/orchestrator.py) calls tools:
   `classify -> retrieve -> generate -> critique -> clarify`
7. Retrieval uses:
   dense search in [src/retrieval/qdrant_store.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/retrieval/qdrant_store.py)
   sparse search in [src/retrieval/bm25_store.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/retrieval/bm25_store.py)
   reranking in [src/retrieval/reranker.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/retrieval/reranker.py)
8. Ticket and message history are persisted in [src/db/ticket_store.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/db/ticket_store.py)

## 3. Dataset Observations

The cleaned KB currently has:

- 9555 rows in `data/processed/kb_clean.csv`
- category distribution:
  Technical: 4366
  Product Inquiry: 3521
  Billing and Payments: 874
  Returns and Exchanges: 471
  Human Resources: 184
  General Inquiry: 139
- high tag coverage:
  `tag_1_clean`: 9013 rows
  `tag_2_clean`: 7242 rows
  `tag_3_clean`: 3882 rows

Important data characteristics:

- the dataset is heavily templated
- many answers are semantically similar
- some category labels are noisy or plainly wrong
- many base KB answers are acknowledgement-style rather than resolution-style
- generated `resolution_chunks.jsonl` entries are often more actionable than the base answers

Implication:

- retrieval quality is constrained not only by search logic but by source-data noise and templated language

## 4. Problems Found

### 4.1 Chat continuity drift

Original behavior:

- follow-ups like “can you explain in simple terms” sometimes jumped to unrelated scenarios
- follow-up handling depended on brittle trigger words and only the last bot reply

Root cause:

- prior context was passed as loose string concatenation
- the original user intent and prior source chunks were not consistently reused

### 4.2 Off-topic questions inside a ticket

Original behavior:

- questions like `who is trump` or `what is the time now` could be treated as part of the active support ticket
- retrieval sometimes returned junk chunks and generation still answered

Root cause:

- short turns inherited the active ticket category
- cache and retrieval could run before the system established whether the turn belonged to the ticket

### 4.3 Accidental ticket creation

Original behavior:

- `/new` followed by a typo like `hu` created a real ticket
- later valid questions were interpreted relative to that bad ticket

Root cause:

- the first non-command free-text message after `/new` immediately opened a ticket
- there was no low-information guard before classification

### 4.4 UI layout issues

Original behavior:

- cache checkbox area rendered as an awkward white box
- short text like `/resolve` wrapped vertically in chat bubbles

Root cause:

- Gradio wrapper elements were not styled correctly
- chat bubble CSS allowed ugly width collapse

### 4.5 Misleading classifier outputs

Original behavior:

- classifier returned `urgency` and `in_scope`
- they looked meaningful but were not truly separate learned signals in the live app

Root cause:

- `urgency` was just keyword heuristics
- `in_scope` was only `confidence > 0.35`

## 5. Fixes Implemented

### 5.1 Better conversational grounding

Updated files:

- [src/ui/ticket_helpers.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/ticket_helpers.py)
- [src/ui/chat_handler.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/chat_handler.py)

Changes:

- recent conversation context now includes a compact transcript instead of only prior user messages
- added helpers to fetch:
  last user message
  last assistant reply
  last assistant source chunks
- follow-up prompts are now anchored to:
  original customer question
  previous assistant reply
  prior retrieved chunks

Outcome:

- follow-ups are much less likely to drift into unrelated topics

### 5.2 Dedicated turn-intent routing layer

Updated files:

- [src/ui/turn_intent.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/turn_intent.py)
- [src/ui/chat_handler.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/chat_handler.py)
- [src/ui/guardrails.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/guardrails.py)

Changes:

- added a dedicated turn-intent layer separate from general guardrails
- routing now explicitly decides:
  `same_issue`
  `new_issue`
  `out_of_scope`
- `chat_handler` now calls the dedicated turn-intent module directly
- the old duplicated routing logic in `guardrails.py` was removed

Outcome:

- ticket continuity is handled by a dedicated routing component rather than scattered heuristics

### 5.3 Low-information input protection

Updated files:

- [src/ui/turn_intent.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/turn_intent.py)
- [src/ui/chat_handler.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/chat_handler.py)

Changes:

- added early low-information detection for accidental or underspecified inputs
- examples:
  `hu`
  `h`
  one-word vague input with no domain signal

Outcome:

- accidental short inputs no longer open tickets or trigger retrieval

### 5.4 Stricter off-topic and new-issue behavior

Updated files:

- [src/ui/turn_intent.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/turn_intent.py)
- [src/ui/chat_handler.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/chat_handler.py)

Changes:

- generic factual questions inside a ticket are blocked as out-of-scope even if the classifier is overconfident
- “new issue” routing is less trigger-happy than before
- same-topic rephrasings like performance issues phrased differently are more likely to stay in the ticket

Outcome:

- fewer false “new issue” prompts
- fewer out-of-ticket junk retrievals

### 5.5 Low-relevance retrieval fallback

Updated files:

- [src/retrieval/reranker.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/retrieval/reranker.py)
- [src/agents/orchestrator.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/agents/orchestrator.py)
- [src/ui/chat_handler.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/chat_handler.py)

Changes:

- reranker now exposes `rerank_score`
- chunks shown in the UI include `rerank_score`
- if retrieved chunks are weak, the bot refuses to answer from them and asks for a clearer issue description

Outcome:

- fewer confident answers built on junk retrieval

### 5.6 Cache behavior cleanup

Updated files:

- [src/agents/orchestrator.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/agents/orchestrator.py)
- [src/retrieval/semantic_cache.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/retrieval/semantic_cache.py)

Changes:

- cached draft extraction was fixed
- cached results now preserve chunks as well as final text

Outcome:

- continuity behaves more consistently when the previous answer came from cache

### 5.7 UI fixes

Updated files:

- [src/ui/layout.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/layout.py)
- [src/ui/styles.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/styles.py)

Changes:

- added a dedicated composer row id
- fixed cache checkbox wrapper styling
- improved textbox min width and row layout
- fixed chat bubble width and word wrapping

Outcome:

- removed the white-box-looking cache wrapper problem
- stopped short messages like `/resolve` from wrapping into narrow vertical bubbles

### 5.8 Removed unused classifier outputs

Updated files:

- [src/models/classifier.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/models/classifier.py)
- [src/tools/classify.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/tools/classify.py)
- [src/tools/clarify.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/tools/clarify.py)
- [configs/config.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/configs/config.py)
- [src/agents/orchestrator.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/agents/orchestrator.py)

Changes:

- removed `urgency`
- removed `in_scope`
- replaced `in_scope` usage with direct confidence thresholds where needed

Outcome:

- classifier outputs now match what the system actually uses

### 5.9 Better KB chunk text

Updated files:

- [data/knowledge_base/prepare_kb_database.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/data/knowledge_base/prepare_kb_database.py)
- [data/knowledge_base/build_kb.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/data/knowledge_base/build_kb.py)
- [scripts/generate_resolution_kb.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/scripts/generate_resolution_kb.py)

Changes:

- chunk `text` now uses:
  `Subject: ...`
  `Customer issue: ...`
  `Resolution: ...`
- this applies to both base KB chunks and generated resolution chunks

Rationale:

- your dataset is highly templated
- the original user query adds important discriminating context that the answer alone often does not contain

Expected outcome after rebuild:

- BM25 should improve because more lexical cues are indexed
- dense retrieval should better separate similar templated answers

### 5.10 Offline model loading for retrieval tools

Updated files:

- [src/retrieval/qdrant_store.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/retrieval/qdrant_store.py)
- [src/retrieval/semantic_cache.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/retrieval/semantic_cache.py)
- [src/retrieval/reranker.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/retrieval/reranker.py)

Changes:

- retrieval components now prefer local transformer snapshot paths from `transformer-models/`
- they only fall back to model names if the local snapshot is missing

Outcome:

- retrieval code is much more reproducible offline

### 5.11 Regression coverage

Added file:

- [scripts/test_chat_routing.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/scripts/test_chat_routing.py)

Current coverage:

- accidental short input does not open a ticket
- greetings are handled before ticket creation
- out-of-scope factual question inside a ticket is blocked
- same-issue performance rephrasing stays in the ticket

Current result:

- `4/4` tests passing

### 5.12 Call scheduling flow

Added files:

- [src/ui/scheduler.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/scheduler.py)

Updated files:

- [src/ui/chat_handler.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/chat_handler.py)
- [src/ui/layout.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/layout.py)
- [src/ui/styles.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/styles.py)

Changes:

- detects chat requests like `schedule a call`
- opens an in-app scheduling panel
- asks the user for:
  email address
  meeting time slot
- builds a confirmation email
- attempts to send the email automatically if SMTP environment variables are configured
- always shows the composed email preview after scheduling

Current email behavior:

- auto-send works only if SMTP config is present:
  `SMTP_HOST`
  `SMTP_PORT`
  `SMTP_USER`
  `SMTP_PASSWORD`
  `SMTP_FROM`
- if SMTP is missing, the UI still prepares the confirmation email draft and shows it to the user

## 6. Retrieval Evaluation Work

Added file:

- [scripts/eval_retrieval_dataset.py](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/scripts/eval_retrieval_dataset.py)

Purpose:

- evaluate retrieval with real KB queries from the dataset
- measure whether the source chunk can be recovered

Metrics:

- `Hit@1`
- `Hit@3`
- `Hit@5`
- `MRR@5`

Status:

- script added
- local transformer loading fixed
- full evaluation is still blocked in the current sandbox because the local Qdrant instance is not accessible from this environment without escalation

## 7. Important Commands

Useful local checks:

- `python3 -m py_compile src/ui/chat_handler.py src/ui/guardrails.py`
- `venv/bin/python -m scripts.test_chat_routing`
- `venv/bin/python -m scripts.eval_retrieval_dataset`

When chunk format changes:

- rebuild the KB and indexes before judging retrieval quality

## 8. Current Known Risks

- source category labels are noisy, which will continue to hurt retrieval and ticket routing
- the classifier still misclassifies some generic questions with moderate confidence
- generated resolution chunks help, but they do not fix mislabeled base data
- retrieval quality cannot be properly judged until the index is rebuilt with the richer chunk text

## 9. Recommended Next Steps

1. Rebuild the KB and indexes using the new chunk text format.
2. Run the dataset-backed retrieval evaluator against the rebuilt index.
3. Inspect misses by category, especially `Technical` and `Product Inquiry`.
4. If retrieval is still weak, add tag-aware boosting and continue cleaning mislabeled categories.
5. Keep extending `scripts/test_chat_routing.py` whenever a new chat failure mode appears.

## 10. Update Policy

This file should be updated whenever we:

- change routing or guardrail logic
- change KB construction or retrieval behavior
- add or remove model outputs
- add tests
- discover a new important failure mode
