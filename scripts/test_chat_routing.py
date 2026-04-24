"""
Focused regression checks for chat guardrails and turn routing.

Run:
    python -m scripts.test_chat_routing
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.ui import guardrails
from src.ui import scheduler
from src.ui.turn_intent import classify_turn_intent


def _assert(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def test_accidental_short_input_does_not_open_ticket():
    verdict = guardrails.low_information_reply("hu", ticket_category=None)
    _assert(verdict is not None, "`hu` should be blocked as low-information input")


def test_greeting_is_handled_before_ticket_creation():
    verdict = guardrails.check("hi how are you", ticket_category=None)
    _assert(verdict is not None, "greeting should be handled by greeting guardrail")
    _assert(verdict["action"] == "greet", "greeting should produce greet action")


def test_out_of_scope_question_inside_ticket_is_blocked():
    route = classify_turn_intent(
        message="what is the time now",
        ticket_category="Technical",
        last_user_message="login fails after the latest update",
    )
    _assert(route["intent"] == "out_of_scope", "time question should be routed out_of_scope")


def test_same_issue_message_stays_in_ticket():
    message = "System issues happening during busy hours"
    route = classify_turn_intent(
        message=message,
        ticket_category="Technical",
        last_user_message="App performance is very slow during peak traffic",
    )
    _assert(route["intent"] == "same_issue", "performance issue should stay in the Technical ticket")


def test_schedule_call_phrase_is_detected():
    _assert(
        scheduler.wants_call_scheduling("Can we schedule a call for this issue?"),
        "schedule a call phrase should be detected",
    )


def main():
    tests = [
        ("accidental short input", test_accidental_short_input_does_not_open_ticket),
        ("greeting handling", test_greeting_is_handled_before_ticket_creation),
        ("out-of-scope inside ticket", test_out_of_scope_question_inside_ticket_is_blocked),
        ("same issue stays in ticket", test_same_issue_message_stays_in_ticket),
        ("schedule call detection", test_schedule_call_phrase_is_detected),
    ]

    print("=" * 64)
    print("CHAT ROUTING REGRESSION TESTS")
    print("=" * 64)

    passed = 0
    for label, fn in tests:
        try:
            fn()
            print(f"[PASS] {label}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {label}: {exc}")

    print("-" * 64)
    print(f"Passed {passed}/{len(tests)} tests")

    if passed != len(tests):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
