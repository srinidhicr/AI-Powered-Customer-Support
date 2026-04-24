"""
Dedicated turn-intent routing layer.

This module decides how a user message should be treated before retrieval:
- low_information
- same_issue
- new_issue
- out_of_scope

It combines the support category classifier with lightweight lexical heuristics.
"""

from __future__ import annotations

from typing import Literal

from src.models.classifier import SupportClassifier

TurnIntent = Literal["same_issue", "new_issue", "out_of_scope"]

DOMAIN_KW: dict[str, set[str]] = {
    "Technical": {
        "wifi", "network", "vpn", "server", "error", "crash", "login",
        "password", "connection", "outage", "bug", "update", "firmware",
        "cloud", "database", "sync", "api", "timeout", "ssl", "install",
        "uninstall", "reboot", "restart", "loading", "slow", "latency",
        "performance", "traffic", "peak", "busy", "system",
    },
    "Billing and Payments": {
        "bill", "billing", "invoice", "charge", "payment", "refund",
        "subscription", "price", "pricing", "cost", "fee", "credit",
        "debit", "transaction", "overcharged", "duplicate", "receipt",
        "paid", "unpaid", "money", "renew",
    },
    "Product Inquiry": {
        "feature", "product", "upgrade", "downgrade", "integration",
        "salesforce", "hubspot", "dashboard", "analytics", "report",
        "export", "import", "documentation", "trial", "demo", "license",
        "spec", "specs", "model", "compatibility",
    },
    "Returns and Exchanges": {
        "return", "exchange", "damaged", "broken", "defective", "package",
        "shipment", "delivery", "order", "replace", "replacement", "wrong",
        "missing",
    },
    "Human Resources": {
        "salary", "leave", "holiday", "hr", "payslip", "contract",
        "onboarding", "offboarding", "policy", "benefit", "promotion",
        "performance",
    },
    "General Inquiry": set(),
}

_GENERIC_QA_PATTERNS: tuple[str, ...] = (
    "who is", "who are you", "what is", "what are", "when is", "where is", "why is",
    "what time is", "time now",
    "tell me about", "define", "meaning of",
)

_LIGHTWORDS: set[str] = {
    "a", "an", "the", "is", "are", "am", "i", "me", "my", "you", "your",
    "to", "for", "of", "in", "on", "at", "it", "this", "that", "do", "does",
    "did", "how", "what", "why", "when", "where", "who", "can", "could",
    "would", "should", "please", "explain", "simple", "terms",
}

_clf = SupportClassifier()


def _content_words(message: str) -> set[str]:
    return {
        word.strip(".,?!:;()[]{}'\"")
        for word in message.lower().split()
        if word.strip(".,?!:;()[]{}'\"") and word not in _LIGHTWORDS
    }


def _lexical_overlap(a: str, b: str) -> float:
    left = _content_words(a)
    right = _content_words(b)
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, min(len(left), len(right)))


def _looks_like_generic_qa(message: str) -> bool:
    lower = message.strip().lower()
    return any(lower.startswith(pattern) for pattern in _GENERIC_QA_PATTERNS)


def low_information_reply(message: str, ticket_category: str | None) -> dict | None:
    lower = message.strip().lower()
    if not lower:
        return None

    words = _content_words(lower)
    all_domain_kws = set().union(*DOMAIN_KW.values())
    has_domain_signal = bool(words & all_domain_kws)

    is_accidental_short = len(lower) <= 3 and not has_domain_signal
    is_one_word_vague = len(lower.split()) == 1 and len(words) <= 1 and not has_domain_signal

    if not (is_accidental_short or is_one_word_vague):
        return None

    if ticket_category:
        reply = (
            "I didn’t get enough detail to help with this ticket yet. "
            "Please describe the issue a bit more, for example what is failing, where it happens, or any error you saw."
        )
    else:
        reply = (
            "That looks too short to open a useful ticket. "
            "Please describe the issue in a few words, for example `login error after update` or `charged twice for subscription`."
        )

    return {"action": "low_information", "reply": reply}


def route_turn(
    message: str,
    ticket_category: str | None,
    classifier_result: dict,
    last_user_message: str = "",
) -> dict:
    if not ticket_category:
        return {"intent": "same_issue"}

    lower = message.strip().lower()
    predicted_category = classifier_result.get("category")
    confidence = float(classifier_result.get("confidence", 0.0))
    overlap = _lexical_overlap(last_user_message, message)
    has_ticket_terms = bool(_content_words(message) & DOMAIN_KW.get(ticket_category, set()))

    if confidence < 0.35 or (_looks_like_generic_qa(lower) and not has_ticket_terms and overlap < 0.15):
        return {
            "intent": "out_of_scope",
            "reply": (
                "This question looks outside the support topics covered by the current ticket. "
                "I can help with product, billing, returns, HR, or technical support issues. "
                "If you want to continue with your support issue, please ask a product-related question."
            ),
        }

    if predicted_category != ticket_category and confidence >= 0.7 and overlap < 0.15 and not has_ticket_terms:
        return {
            "intent": "new_issue",
            "reply": (
                f"This looks like a new **{predicted_category}** issue rather than part of your current "
                f"**{ticket_category}** ticket.\n\nPlease use **`/new`** to open it as a separate ticket, "
                "or rephrase if it is still about the current issue."
            ),
        }

    return {"intent": "same_issue"}


def classify_turn_intent(
    message: str,
    ticket_category: str | None,
    last_user_message: str = "",
) -> dict:
    """
    Run the dedicated turn-intent layer.

    Returns classifier output plus routing decision:
    {
      "category": ...,
      "confidence": ...,
      "latency_ms": ...,
      "intent": "same_issue" | "new_issue" | "out_of_scope",
      "reply": "...",  # present only for non-same_issue routes
    }
    """
    clf = _clf.predict(subject="", body=message, tags=None)
    route = route_turn(
        message=message,
        ticket_category=ticket_category,
        classifier_result=clf,
        last_user_message=last_user_message,
    )
    return {**clf, **route}
