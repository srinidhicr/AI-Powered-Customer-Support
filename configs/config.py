# configs/config.py

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # LLM
    google_api_key: str  = os.getenv("GOOGLE_API_KEY", "")
    llm_model:      str  = "gpt-4.1-mini" # "gemini-2.5-flash"
    openai_api_key: str  = os.getenv("OPENAI_API_KEY", "")

    # Classifier
    classifier_path: str   = "src/models/saved/best_classifier.pkl"

    # Qdrant
    qdrant_url:       str = os.getenv("QDRANT_URL", "localhost")
    qdrant_port:      int = 6333
    collection_name:  str = "support_kb"

    # Retrieval
    top_k_dense:  int = 20
    top_k_sparse: int = 20
    top_k_final:  int = 5

    # Critique
    critique_threshold: float = 0.60
    max_retries:        int   = 1

    # LangSmith
    langsmith_api_key: str = os.getenv("LANGSMITH_API_KEY", "")
    langsmith_project: str = "customer-support-agent"

    # ── NEW ──
    skip_critique_rerank:    float = 4.0   # skip critique if best rerank score exceeds this
    skip_rewrite_confidence: float = 0.75  # skip query rewrite if classifier confidence exceeds this

config = Config()
