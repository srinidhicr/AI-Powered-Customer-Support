# src/tools/classify.py

import os, sys, json
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from langchain.tools import tool
from pydantic import BaseModel, Field
from src.models.classifier import SupportClassifier

_clf = SupportClassifier()


# ── Pydantic input schema ─────────────────────────────────────────────────────

class ClassifyInput(BaseModel):
    query: str = Field(
        description="The raw customer query text to classify into a support category."
    )


# ── Tool ──────────────────────────────────────────────────────────────────────

@tool(args_schema=ClassifyInput)
def classify(query: str) -> str:
    """Classifies a customer support query into a support category.
    Returns category and confidence score (0-1).
    ALWAYS call this tool first before any other tool.
    Returns a JSON string with keys: category, confidence, latency_ms.
    """
    result = _clf.predict(subject='', body=query, tags=None)
    print(f"[Classifier] category={result['category']} | confidence={result['confidence']}")
    return json.dumps(result)
