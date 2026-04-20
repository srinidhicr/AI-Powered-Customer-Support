# scripts/test_pipeline.py
#
# Quick CLI test for the full pipeline — no UI needed.
# Run: python -m scripts.test_pipeline
#
# Options:
#   --query "your query here"   → single query mode
#   --batch                     → run all built-in test queries
#   --cache-test                → run the semantic cache pair test
#   --no-cache                  → disable cache for this run

import sys, os, json, argparse, textwrap
from src.agents.orchestrator import get_final_draft

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# ── built-in test queries ─────────────────────────────────────────────────────

TEST_QUERIES = {
    "Technical - network outage": (
        "My laptop cannot connect to the office WiFi after the latest Windows update. "
        "I've restarted both the router and my device but the issue persists."
    ),
    "Technical - cloud outage": (
        "We're experiencing a full outage on our cloud storage platform. "
        "Multiple team members cannot access files and the dashboard won't load."
    ),
    "Technical - VPN": (
        "The VPN disconnects every 20 minutes on macOS. "
        "We've reinstalled the client twice and the problem continues."
    ),
    "Billing - duplicate charge": (
        "I was charged twice for my subscription this month. "
        "The duplicate transaction appeared on the 14th. Please advise on how to get a refund."
    ),
    "Billing - plan question": (
        "Can you explain the difference between the Standard and Enterprise billing plans? "
        "We're a team of 50 and need to understand volume discounts."
    ),
    "Product Inquiry - integration": (
        "Does your analytics dashboard integrate with Salesforce and HubSpot? "
        "We need bidirectional data sync."
    ),
    "Returns - stress test": (
        "I'd like to return a device I purchased 3 weeks ago. "
        "The build quality was not as described on the website."
    ),
    "Low confidence - should clarify": (
        "Something is wrong and I need help urgently."
    ),
    "Ambiguous - should clarify": (
        "Hi, I have a question about my account."
    ),
}

# Cache test pair — send first, then second; second should be a cache hit
CACHE_TEST_QUERIES = {
    "cache_seed": (
        "My payment failed but I can see the amount was deducted from my bank account."
    ),
    "cache_hit_candidate": (
        "The payment didn't go through but money has already left my account. What do I do?"
    ),
}


# ── display helpers ───────────────────────────────────────────────────────────

def print_separator(title: str = ''):
    width = 70
    if title:
        pad = (width - len(title) - 2) // 2
        print('\n' + '─' * pad + f' {title} ' + '─' * pad)
    else:
        print('\n' + '─' * width)


def print_result(label: str, query: str, result: dict):
    print_separator(label)
    print(f"Query:\n  {textwrap.fill(query, width=66, subsequent_indent='  ')}")

    if result.get("cache_hit"):
        print("\n[CACHE HIT ✓]")
        draft = result.get("cached_result", {}).get("final_draft", "")
        print(f"\nCached Draft:\n{textwrap.fill(draft, width=66, initial_indent='  ', subsequent_indent='  ')}")
        return

    # Extract final message from agent
    final = get_final_draft(result)
    print("RESULT: ", result)
    print("FINAL ", final)

    # Try to pretty-print if it's JSON
    try:
        print(final)
        parsed = json.loads(final)
        print(parsed)
        print("CAME HEREE")
        if "clarifying_question" in parsed:
            print("HERE")
            print(f"\n[CLARIFY] {parsed['clarifying_question']}")
            print(f"Reason: {parsed.get('reason', '')}")
            print("done 1")
        elif "draft_response" in parsed:
            print("OR HERE")
            print(f"\nDraft Response:")
            print(textwrap.fill(
                parsed["draft_response"], width=66,
                initial_indent='  ', subsequent_indent='  '
            ))
            print(f"\nConfidence : {parsed.get('confidence', 'n/a')}")
            print(f"Caveats    : {parsed.get('caveats', 'none')}")
        else:
            print(f"\nOutput:\n{json.dumps(parsed, indent=2)}")
    except (json.JSONDecodeError, TypeError):
        print("ERRORRR")
        # Plain text output — just print it
        print(f"\nFinal Output:\n{textwrap.fill(final, width=66, initial_indent='  ', subsequent_indent='  ')}")


# ── runners ───────────────────────────────────────────────────────────────────

def run_single(query: str, use_cache: bool = True):
    from src.agents.orchestrator import run
    print_separator("SINGLE QUERY TEST")
    print(f"Query: {query}\n")
    result = run(query, use_cache=use_cache)
    print_result("Result", query, result)


def run_batch(use_cache: bool = True):
    from src.agents.orchestrator import run
    print_separator("BATCH TEST")
    print(f"Running {len(TEST_QUERIES)} queries...\n")

    passed = 0
    for label, query in TEST_QUERIES.items():
        try:
            result = run(query, use_cache=use_cache)
            print_result(label, query, result)
            passed += 1
        except Exception as e:
            print_separator(f"ERROR — {label}")
            print(f"  {e}")

    print_separator("SUMMARY")
    print(f"Completed: {passed}/{len(TEST_QUERIES)}")


def run_cache_test():
    from src.agents.orchestrator import run
    from src.retrieval.semantic_cache import clear

    print_separator("SEMANTIC CACHE TEST")
    print("Step 1: Clearing cache to start fresh...")
    clear()

    seed_label  = "cache_seed"
    hit_label   = "cache_hit_candidate"
    seed_query  = CACHE_TEST_QUERIES[seed_label]
    hit_query   = CACHE_TEST_QUERIES[hit_label]

    print(f"\nStep 2: Running SEED query (expect no cache hit)...")
    result1 = run(seed_query, use_cache=True)
    print_result(seed_label, seed_query, result1)
    print(f"\n  cache_hit = {result1.get('cache_hit')}")

    print(f"\nStep 3: Running SIMILAR query (expect cache hit)...")
    result2 = run(hit_query, use_cache=True)
    print_result(hit_label, hit_query, result2)
    print(f"\n  cache_hit = {result2.get('cache_hit')}")

    if result2.get("cache_hit"):
        print("\n[PASS] Cache hit detected correctly ✓")
    else:
        print("\n[FAIL] No cache hit — consider lowering SIMILARITY_THRESHOLD in semantic_cache.py")
        print("       Current default is 0.92. Try 0.88.")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test the customer support pipeline")
    parser.add_argument('--query',      type=str, help='Single query to test')
    parser.add_argument('--batch',      action='store_true', help='Run all built-in test queries')
    parser.add_argument('--cache-test', action='store_true', help='Run semantic cache pair test')
    parser.add_argument('--no-cache',   action='store_true', help='Disable semantic cache')
    args = parser.parse_args()

    use_cache = not args.no_cache

    if args.query:
        run_single(args.query, use_cache=use_cache)
    elif args.batch:
        run_batch(use_cache=use_cache)
    elif args.cache_test:
        run_cache_test()
    else:
        # Default: run a quick smoke test with 2 queries
        print_separator("SMOKE TEST  (pass --batch for all queries)")
        from src.agents.orchestrator import run

        smoke = [
            ("Technical", "We're having a complete network outage. All WiFi connections dropped after a router firmware update."),
            ("Clarify",   "Something is wrong and I need help urgently."),
        ]
        for label, q in smoke:
            result = run(q, use_cache=use_cache)
            print_result(label, q, result)


if __name__ == '__main__':
    main()