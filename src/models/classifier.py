# src/models/classifier.py
# Wrapper used at inference time — loaded once, called many times

import joblib
import re
import time
from src.preprocessing.text_cleaner import (
    clean_body, normalize_tags, remove_leakage
)

MODEL_PATH = 'src/models/saved/best_classifier.pkl'

URGENCY_HIGH   = [
    'urgent', 'critical', 'emergency', 'fraud',
    'breach', 'down', 'outage', 'immediately'
]
URGENCY_MEDIUM = [
    'problem', 'issue', 'error', 'not working', 'failed', 'delay'
]


class SupportClassifier:
    def __init__(self, model_path: str = MODEL_PATH):
        self._model = joblib.load(model_path)
        self.classes_ = list(self._model.classes_)

    def _build_input(self, subject: str, body: str, tags: dict = None) -> str:
        # 1. Clean the body
        body_clean = clean_body(body)

        # 2. Replicate the 'text' logic from training: "Subject - Body"
        subject_part = str(subject or '').strip()
        full_text = f"{subject_part} - {body_clean}".strip()

        # 3. Handle the leading dash removal exactly like training
        full_text = re.sub(r'^\s*-\s*', '', full_text)

        # 4. Handle Tags
        tags_text = ''
        if tags:
            tag_list = normalize_tags(tags)
            tag_clean = remove_leakage(tag_list)
            tags_text = ' '.join(tag_clean)

        # 5. Join with a single space (ensure no double spaces if tags_text is empty)
        final = f"{full_text} {tags_text}".strip()
        return final

    def predict(
        self,
        subject: str,
        body: str,
        tags: dict = None
    ) -> dict:
        text      = self._build_input(subject, body, tags)
        start     = time.time()
        pred      = self._model.predict([text])[0]
        proba     = self._model.predict_proba([text])[0].max()
        latency   = round((time.time() - start) * 1000, 2)

        q = (body or '').lower()
        if any(w in q for w in URGENCY_HIGH):
            urgency = 'high'
        elif any(w in q for w in URGENCY_MEDIUM):
            urgency = 'medium'
        else:
            urgency = 'low'

        return {
            'category'  : pred,
            'confidence': round(float(proba), 3),
            'urgency'   : urgency,
            'in_scope'  : float(proba) > 0.35,
            'latency_ms': latency
        }