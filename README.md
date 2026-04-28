# AI-Powered Customer Support

An AI-powered customer support copilot built with Gradio, LangGraph, OpenAI, Qdrant, BM25, and a custom ticketing UI. The project combines category classification, turn-intent routing, hybrid retrieval, semantic caching, and grounded response generation for support-style conversations.

## What This Project Does

- Opens and manages support tickets in a chat UI
- Classifies user issues into support categories
- Routes each turn as same-issue, new-issue, out-of-scope, follow-up, or low-information
- Retrieves relevant support knowledge with dense + sparse + reranked retrieval
- Generates grounded support responses from the knowledge base
- Stores ticket history and retrieved chunks in SQLite
- Supports scheduling a support call with email confirmation
- Includes evaluation scripts for retrieval metrics

## Core Features

- `Gradio` chat interface with ticket sidebar and source chunk display
- `LangGraph` orchestrator with tools for classify, retrieve, generate, critique, and clarify
- `Qdrant` dense vector search
- `BM25` sparse retrieval
- Cross-encoder reranking
- Semantic cache for repeated or similar questions
- Follow-up clarification flow that avoids unnecessary re-retrieval
- Resolution-chunk generation pipeline to enrich templated support data
- Retrieval evaluation and RAGAS evaluation scripts

## Project Structure

```text
.
├── configs/
│   └── config.py
├── data/
│   ├── raw/
│   │   └── cust_support_ticket.csv
│   ├── processed/
│   │   └── kb_clean.csv
│   ├── knowledge_base/
│   │   ├── kb_chunks.jsonl
│   │   ├── resolution_chunks.jsonl
│   │   ├── kb_chunks_merged.jsonl
│   │   └── bm25_index.pkl
│   └── tickets.db
├── scripts/
│   ├── prepare_kb_database.py
│   ├── generate_resolution_kb.py
│   ├── build_kb_script.py
│   ├── eval_main.py
│   ├── test_chat_routing.py
│   └── test_pipeline.py
├── src/
│   ├── agents/
│   │   └── orchestrator.py
│   ├── db/
│   │   └── ticket_store.py
│   ├── models/
│   │   ├── saved/
│   │   │   └── best_classifier.pkl
│   │   ├── classifier.py
│   │   └── train_classifier.py
│   ├── preprocessing/
│   │   └── text_cleaner.py
│   ├── retrieval/
│   │   ├── qdrant_store.py
│   │   ├── bm25_store.py
│   │   ├── reranker.py
│   │   └── semantic_cache.py
│   ├── tools/
│   │   ├── classify.py
│   │   ├── retrieve.py
│   │   ├── generate.py
│   │   ├── critique.py
│   │   └── clarify.py
│   └── ui/
│       ├── components/
│       │     ├── chunks.py
│       │     └── commands.py
│       ├── app.py
│       ├── layout.py
│       ├── chat_handler.py
│       ├── turn_intent.py
│       ├── guardrails.py
│       ├── scheduler.py
│       ├── styles.py
│       └── ticket_helpers.py
├── PROJECT_REPORT.md
├── req.txt
└── qdrant
```

## Architecture Overview
![Query flow in architecture](report/flowcharts/Full Query Flow.png)

### 1. UI and Ticketing

- The main app entrypoint is [`src/ui/app.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/app.py).
- The Gradio layout is defined in [`src/ui/layout.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/layout.py).
- Chat flow, routing, ticket creation, follow-ups, and scheduler actions are handled in [`src/ui/chat_handler.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/chat_handler.py).
- Ticket and message persistence is stored in SQLite via [`src/db/ticket_store.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/db/ticket_store.py).

### 2. Routing Layer

Before retrieval or generation, each turn is filtered through lightweight routing:

- slash commands like `/help`, `/new`, `/status`, `/resolve`
- low-information or accidental-short-message checks
- greeting / noise / out-of-scope checks
- call-scheduling detection
- follow-up detection
- turn-intent routing in [`src/ui/turn_intent.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/turn_intent.py)

This keeps the orchestrator focused on real support turns instead of handling every message.

### 3. Orchestrator

The support pipeline lives in [`src/agents/orchestrator.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/agents/orchestrator.py).

It uses a LangGraph ReAct agent with these tools:

- `classify`
- `retrieve`
- `generate`
- `critique`
- `clarify`

For certain paths, the UI intentionally bypasses the full tool loop:

- slash commands
- low-information turns
- greetings / simple guardrail replies
- follow-up clarification replies
- meeting scheduling flow

### 4. Retrieval

Hybrid retrieval combines:

- Dense retrieval in [`src/retrieval/qdrant_store.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/retrieval/qdrant_store.py)
- Sparse retrieval in [`src/retrieval/bm25_store.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/retrieval/bm25_store.py)
- Cross-encoder reranking in [`src/retrieval/reranker.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/retrieval/reranker.py)
- Semantic cache in [`src/retrieval/semantic_cache.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/retrieval/semantic_cache.py)

### 5. Knowledge Base Pipeline

The KB build process is:

1. Clean the raw ticket dataset
2. Generate structured knowledge chunks
3. Optionally generate improved resolution chunks
4. Merge original and generated chunks
5. Build Qdrant dense index
6. Build BM25 sparse index

Important scripts:

- [`scripts/prepare_kb_database.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/scripts/prepare_kb_database.py)
- [`scripts/generate_resolution_kb.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/scripts/generate_resolution_kb.py)
- [`scripts/build_kb_script.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/scripts/build_kb_script.py)

## Setup

### Prerequisites

- Python 3.11+ recommended
- `pip`
- An OpenAI API key
- A running Qdrant instance on `localhost:6333`, or custom `QDRANT_URL` / `QDRANT_PORT`

Optional:

- LangSmith API key for tracing
- SMTP credentials for meeting confirmation emails

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r req.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root.

Example:

```env
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=
LANGSMITH_API_KEY=
QDRANT_URL=localhost
QDRANT_PORT=6333

# Optional SMTP settings for scheduling flow
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
SMTP_FROM=
```

### 4. Start Qdrant

This project expects a Qdrant server to be available.

If you already have Qdrant installed locally, run it on port `6333`.

If you prefer Docker:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

This repository also contains a local `qdrant` executable at [`qdrant`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/qdrant), which may be useful if you prefer a bundled local binary.

## Build the Knowledge Base

### Recommended full build

```bash
python -m scripts.build_kb_script
```

This will:

- preprocess the raw dataset into cleaned chunks
- merge original chunks with generated resolution chunks if present
- index the merged chunks into Qdrant
- build the BM25 sparse index

### Manual build steps

If you want to run the steps individually:

#### 1. Prepare the cleaned KB

```bash
python -m scripts.prepare_kb_database
```

Outputs:

- `data/processed/kb_clean.csv`
- `data/knowledge_base/kb_chunks.jsonl`

#### 2. Generate resolution chunks

```bash
python -m scripts.generate_resolution_kb
```

Output:

- `data/knowledge_base/resolution_chunks.jsonl`

This step uses OpenAI to convert templated acknowledgement-style answers into more actionable resolution-oriented support chunks.

#### 3. Merge and index everything

```bash
python -m scripts.build_kb_script
```

## Run the App

Start the Gradio UI:

```bash
python src/ui/app.py
```

The app launches on:

- `http://0.0.0.0:7860`

Open that in your browser and start chatting.

## Supported Chat Commands

- `/help` — show command menu
- `/new` — start a new ticket
- `/status` — show current ticket info
- `/resolve` — mark current ticket resolved

## Scheduling Flow

If a user says things like:

- `schedule a call`
- `schedule a meeting`
- `book a call`

the UI opens a scheduling panel.

The user can:

- enter an email address
- choose a time slot
- generate a confirmation email
- send the confirmation automatically if SMTP is configured

Scheduler logic lives in [`src/ui/scheduler.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/scheduler.py).

## Evaluation

This project keeps two main evaluation entrypoints.

### 1. Standard retrieval evaluation

Run:

```bash
python -m scripts.eval_main
```

What it measures:

- `Hit@1`
- `Hit@3`
- `Hit@5`
- `Precision@1`
- `Precision@3`
- `Precision@5`
- `MRR@5`

This script evaluates retrieval by checking whether the original source chunk is recovered for sampled queries from the KB.

Note: this benchmark is useful, but optimistic for heavily templated datasets because the original query text is also present in the indexed chunk text.


## Testing and Debugging

### Pipeline smoke test

```bash
python -m scripts.test_pipeline
```

Useful modes:

```bash
python -m scripts.test_pipeline --batch
python -m scripts.test_pipeline --cache-test
python -m scripts.test_pipeline --query "How do I debug performance issues in analytics software?"
```

## Important Files

### App

- [`src/ui/app.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/app.py) — app entrypoint
- [`src/ui/layout.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/layout.py) — Gradio layout
- [`src/ui/chat_handler.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/chat_handler.py) — main chat workflow
- [`src/ui/turn_intent.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/ui/turn_intent.py) — turn-level routing

### Retrieval and generation

- [`src/agents/orchestrator.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/agents/orchestrator.py) — LangGraph orchestration
- [`src/retrieval/qdrant_store.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/retrieval/qdrant_store.py) — dense retrieval
- [`src/retrieval/bm25_store.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/retrieval/bm25_store.py) — sparse retrieval
- [`src/retrieval/reranker.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/retrieval/reranker.py) — reranking
- [`src/retrieval/semantic_cache.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/src/retrieval/semantic_cache.py) — semantic caching

### Data and preprocessing

- [`data/raw/cust_support_ticket.csv`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/data/raw/cust_support_ticket.csv) — raw dataset
- [`scripts/prepare_kb_database.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/scripts/prepare_kb_database.py) — KB cleaning and chunk preparation
- [`scripts/generate_resolution_kb.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/scripts/generate_resolution_kb.py) — resolution chunk generation
- [`scripts/build_kb_script.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/scripts/build_kb_script.py) — complete KB build pipeline

### Evaluation

- [`scripts/eval_main.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/scripts/eval_main.py)
- [`scripts/eval_ragas.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/scripts/eval_ragas.py)
- [`scripts/test_pipeline.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/scripts/test_pipeline.py)
- [`scripts/test_chat_routing.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/scripts/test_chat_routing.py)

## Configuration Notes

Key config values are defined in [`configs/config.py`](/Users/srinidhicr/PROJECTS/ai-powered-customer-support/configs/config.py):

- `llm_model = "gpt-4.1-mini"`
- `collection_name = "support_kb"`
- `top_k_dense = 20`
- `top_k_sparse = 20`
- `top_k_final = 5`
- `critique_threshold = 0.60`
- `max_retries = 1`

## Data Notes

This dataset is heavily templated. Because of that:

- many queries are semantically similar
- many base answers are acknowledgement-style rather than resolution-style
- sparse retrieval can look unusually strong on self-retrieval benchmarks
- generated resolution chunks can improve response usefulness

The current chunk format includes:

- `subject`
- `query`
- `answer`
- `tags`
- combined `text` used for retrieval, including subject + customer issue + resolution

## Recommended First Run

If you are setting this project up from scratch, the shortest path is:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r req.txt
python -m scripts.build_kb_script
python src/ui/app.py
```

Before running those commands, make sure:

- `OPENAI_API_KEY` is set
- Qdrant is running on `localhost:6333` or your configured host/port
