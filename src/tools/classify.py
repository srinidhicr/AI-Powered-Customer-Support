# src/tools/classify.py

import os, sys, json
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from langchain.tools import tool
from src.models.classifier import SupportClassifier

_clf = SupportClassifier()

@tool
def classify(query: str) -> str:
    """Classifies a customer support query into a support category.
    Returns category, confidence score (0-1), urgency level, and in_scope flag.
    ALWAYS call this tool first before any other tool.

    Args:
        query: The raw customer query text.

    Returns:
        JSON string with keys: category, confidence, urgency, in_scope, latency_ms
    """
    result = _clf.predict(subject='', body=query, tags=None)
    print(f"[Classifier] category={result['category']} | confidence={result['confidence']}")
    return json.dumps(result)