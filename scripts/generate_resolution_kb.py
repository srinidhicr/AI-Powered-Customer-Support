# scripts/generate_resolution_kb.py
"""
This script generates resolution chunks to enrich the knowledge base
"""
import sys, os, json, time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from google import genai
from configs.config import config
import pandas as pd
from openai import OpenAI

INPUT_CSV    = 'data/processed/kb_clean.csv'
OUTPUT_JSONL = 'data/knowledge_base/resolution_chunks.jsonl'
PROGRESS_FILE = 'data/knowledge_base/.generation_progress.json'

#client = genai.Client(api_key=config.openai_api_key)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

ACKNOWLEDGEMENT_PATTERNS = [
    'please provide', 'could you please', 'kindly provide',
    'thank you for reaching out', 'we are working',
    'our team is investigating', 'we will look into',
    'to assist you further', 'to help us assist',
    'please describe', 'please specify', 'please confirm',
    'please share', 'please inform', 'please let'
]

def is_acknowledgement(text: str) -> bool:
    t = str(text).lower()
    return sum(1 for p in ACKNOWLEDGEMENT_PATTERNS if p in t) >= 2


def generate_resolution(subject: str, query: str, category: str,
                        tags: list, bad_answer: str) -> str:
    tags_str = ', '.join(tags) if tags else 'none'
    prompt = f"""
        You are a technical knowledge base writer for a customer support system.

        A customer submitted this support ticket:
        Subject: {subject}
        Category: {category}
        Tags: {tags_str}
        Query: {query}

        The original support response was unhelpful:
        "{bad_answer}"

        Write a RESOLUTION response that:
        1. Briefly acknowledges the issue (1 sentence max)
        2. Provides 3-5 concrete, actionable troubleshooting steps specific to this issue
        3. States what to do if those steps don't work (escalation path)
        4. Is 100-200 words total

        Be specific. Use numbered steps. Do NOT say "please provide more information" as the main response.
        Write only the response text, no preamble.
    """
    """
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            if '429' in str(e) or 'RESOURCE_EXHAUSTED' in str(e):
                wait = 60 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Error: {e}")
                return ""
    return ""
    """
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                time.sleep(30)
            else:
                raise
    return ""


def load_progress() -> set:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return set(json.load(f).get('done_ids', []))
    return set()


def save_progress(done_ids: set):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({'done_ids': list(done_ids)}, f)


#multiple queries
def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} KB entries")

    # Filter to acknowledgement-only answers — these need upgrading
    df['is_ack'] = df['answer'].apply(is_acknowledgement)
    to_upgrade = df[df['is_ack']].copy()
    print(f"Found {len(to_upgrade)} acknowledgement-only answers to upgrade")

    # Limit to most impactful categories for demo
    # Focus on Technical and Billing which have most volume
    priority_cats = ['Technical', 'Billing and Payments', 'Product Inquiry', 'Returns and Exchanges']
    to_upgrade = to_upgrade[to_upgrade['category'].isin(priority_cats)]

    # Sample up to 200 per category to stay within quota
    # 200 entries × ~3 LLM calls overhead = manageable
    priority_cats = ['Technical', 'Billing and Payments', 'Product Inquiry', 'Returns and Exchanges']
    to_upgrade = to_upgrade[to_upgrade['category'].isin(priority_cats)].copy()

    # Sample manually per category without groupby
    frames = []
    for cat in priority_cats:
        cat_df = to_upgrade[to_upgrade['category'] == cat]
        n = min(len(cat_df), 50)
        frames.append(cat_df.sample(n, random_state=42))

    sampled = pd.concat(frames).reset_index(drop=True)

    print(f"Sampled {len(sampled)} entries")
    print(f"Columns: {list(sampled.columns)}")
    print(f"Category distribution:\n{sampled['category'].value_counts()}")

    done_ids = load_progress()
    print(f"Already done: {len(done_ids)}")

    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

    # Open in append mode — safe to resume
    with open(OUTPUT_JSONL, 'a') as f:
        for i, row in sampled.iterrows():
            row_id = str(i)
            if row_id in done_ids:
                continue

            tags = [
            str(t).strip()
            for t in [
                row.get('tag_1_clean'),
                row.get('tag_2_clean'),
                row.get('tag_3_clean')
            ]
            if pd.notna(t) and str(t).strip() != ''
            ]

            print(f"[{len(done_ids)+1}/{len(sampled)}] {str(row.get('subject', ''))[:50]}...")
            resolution = generate_resolution(
                subject   = str(row.get('subject', '')),
                query     = str(row.get('query', '')),
                category  = str(row['category']),
                tags      = tags,
                bad_answer= str(row['answer'])
            )

            if resolution:
                subject_str = str(row.get('subject', '') or '').strip()
                if subject_str.lower() == 'nan':
                    subject_str = ''
                query_str = str(row.get('query', '') or '').strip()
                text_parts = []
                if subject_str:
                    text_parts.append(f"Subject: {subject_str}")
                if query_str:
                    text_parts.append(f"Customer issue: {query_str}")
                text_parts.append(f"Resolution: {resolution}")

                chunk = {
                    "id"      : f"resolution_{i}",
                    "text"    : "\n".join(text_parts),
                    "answer"  : resolution,
                    "query"   : query_str,
                    "subject" : subject_str,
                    "category": str(row['category']),
                    "priority": str(row.get('priority', '') or ''),
                    "tags"    : tags,
                    "source"  : "generated_resolution"
                }
                f.write(json.dumps(chunk) + '\n')
                done_ids.add(row_id)
                save_progress(done_ids)
                time.sleep(4)

    print(f"\nDone. Generated {len(done_ids)} resolution chunks.")
    print(f"Output: {OUTPUT_JSONL}")

"""
def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} KB entries")

    df['is_ack'] = df['answer'].apply(is_acknowledgement)
    to_upgrade = df[df['is_ack']].copy()
    print(f"Found {len(to_upgrade)} acknowledgement-only answers to upgrade")

    priority_cats = ['Technical', 'Billing and Payments', 'Product Inquiry', 'Returns and Exchanges']
    to_upgrade = to_upgrade[to_upgrade['category'].isin(priority_cats)]

    # ── PREVIEW MODE: just 5 samples, no file writing ────────────────────
    preview = to_upgrade.sample(5, random_state=42)

    print("\n" + "="*60)
    print("PREVIEW — 5 sample resolutions (not saved)")
    print("="*60)

    for i, (_, row) in enumerate(preview.iterrows(), 1):
        tags = [
            str(t).strip()
            for t in [
                row.get('tag_1_clean'),
                row.get('tag_2_clean'),
                row.get('tag_3_clean')
            ]
            if pd.notna(t) and str(t).strip() != ''
        ]

        print(f"\n[{i}/5] Subject: {row['subject']}")
        print(f"      Category: {row['category']}")
        print(f"      Tags: {tags}")
        print(f"\n  ORIGINAL ANSWER:")
        print(f"  {row['answer'][:200]}...")
        print(f"\n  GENERATED RESOLUTION:")

        resolution = generate_resolution(
            subject   = str(row.get('subject', '')),
            query     = str(row.get('query', '')),
            category  = str(row['category']),
            tags      = tags,
            bad_answer= str(row['answer'])
        )

        if resolution:
            # Print with indentation for readability
            for line in resolution.split('\n'):
                print(f"  {line}")
        else:
            print("  [FAILED TO GENERATE]")

        print("\n" + "-"*60)
        time.sleep(4)

    print("\nPreview complete. If quality looks good, remove the preview")
    print("section and run the full generation.")
"""     

if __name__ == '__main__':
    main()
