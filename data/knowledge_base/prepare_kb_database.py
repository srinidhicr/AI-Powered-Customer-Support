# data/knowledge_base/prepare_kb_dataset.py

import os, sys, re, json
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = 'data/raw/cust_support_ticket.csv'
CSV_OUT   = 'data/processed/kb_clean.csv'
JSONL_OUT = 'data/knowledge_base/kb_chunks.jsonl'

CATEGORY_MERGE = {
    'IT Support'                     : 'Technical',
    'Technical Support'              : 'Technical',
    'Service Outages and Maintenance': 'Technical',
    'Product Support'                : 'Product Inquiry',
    'Customer Service'               : 'Product Inquiry',
    'Sales and Pre-Sales'            : 'Product Inquiry',
}

LEAKAGE_KEYWORDS = [
    'billing', 'payment', 'technical', 'tech', 'support',
    'sales', 'return', 'exchange', 'hr', 'human resources', 'it',
    'customer service', 'customer support', 'product support',
    'technical support', 'service', 'issue', 'request'
]

STRIP_PATTERNS = [
    r'dear\s+(customer\s+support(\s+team)?|support\s+team|support|team)[,\s]*',
    r'^(support\s+team|customer\s+support(\s+team)?)[,\s]*',
    r'i\s+hope\s+this\s+message\s+(reaches|finds)\s+you\s+(well|in\s+good\s+health)[.\s,]*',
    r'i\s+am\s+(writing|reaching\s+out|submitting\s+a\s+report)\s+to\s+',
    r'thank\s+you\s+(very\s+much\s+)?for\s+your\s+(assistance|help|support)[.\s]*',
    r'best\s+regards[,.\s]*',
    r'kind\s+regards[,.\s]*',
    r'yours\s+(sincerely|faithfully)[,.\s]*',
    r'i\s+look\s+forward\s+to\s+your\s+(prompt\s+)?repl(y|ies)[.\s]*',
]


def clean_query(text: str) -> str:
    if pd.isna(text):
        return ''
    text = str(text)
    text = text.replace("\\n", " ").replace("\n", " ")
    for pattern in STRIP_PATTERNS:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_tags(row) -> list:
    TAG_COLS = ['tag_1', 'tag_2', 'tag_3']
    collected = []
    for col in TAG_COLS:
        val = row.get(col, '')
        if pd.isna(val) or str(val).strip() == '':
            continue
        parts = [p.strip().lower() for p in str(val).split(',') if p.strip()]
        collected.extend(parts)
    seen, unique = set(), []
    for tag in collected:
        if tag not in seen:
            seen.add(tag)
            unique.append(tag)
    clean = [t for t in unique if not any(kw in t for kw in LEAKAGE_KEYWORDS)]
    return clean[:3]


def clean_answer(text: str) -> str:
    if pd.isna(text):
        return ''
    text = str(text)
    text = text.replace("\\n", " ").replace("\n", " ")
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'<name>', 'the customer', text, flags=re.IGNORECASE)
    text = re.sub(r'<tel_num>', 'our support hotline', text, flags=re.IGNORECASE)
    text = re.sub(r'<[a-z_]+>', '[contact details]', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    incomplete_endings = ['we recommend...', 'please note that...', 'additionally,...']
    if any(text.lower().endswith(e) for e in incomplete_endings):
        return ''
    return text


def deduplicate_answers(df: pd.DataFrame, threshold: float = 0.75) -> pd.DataFrame:
    print("\nDeduplicating answers...")
    answers      = df['answer_clean'].tolist()
    vectorizer   = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(answers)
    keep, dropped = [], 0
    for i in range(len(answers)):
        if i == 0:
            keep.append(i)
            continue
        sims = cosine_similarity(tfidf_matrix[i], tfidf_matrix[keep])[0]
        if sims.max() < threshold:
            keep.append(i)
        else:
            dropped += 1
    print(f"  Kept {len(keep)} unique answers, dropped {dropped} near-duplicates")
    return df.iloc[keep].reset_index(drop=True)


def prepare():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Raw rows: {len(df)}")

    df = df[df['language'] == 'en'].copy()
    print(f"  After language filter: {len(df)}")

    df = df.dropna(subset=['answer'])
    df = df[df['answer'].str.strip() != '']
    print(f"  After dropping empty answers: {len(df)}")

    # category — merged (queue is only used here to derive category, then dropped)
    df['category'] = df['queue'].replace(CATEGORY_MERGE)

    df['query_clean']  = df['body'].apply(clean_query)
    df['answer_clean'] = df['answer'].apply(clean_answer)

    df['tags']        = df.apply(normalize_tags, axis=1)
    df['tag_1_clean'] = df['tags'].apply(lambda t: t[0] if len(t) > 0 else '')
    df['tag_2_clean'] = df['tags'].apply(lambda t: t[1] if len(t) > 1 else '')
    df['tag_3_clean'] = df['tags'].apply(lambda t: t[2] if len(t) > 2 else '')

    df['answer_len'] = df['answer_clean'].str.split().str.len()
    before = len(df)
    df = df[df['answer_len'] >= 10].reset_index(drop=True)
    print(f"  After dropping short answers: {len(df)}  (removed {before - len(df)})")

    df = deduplicate_answers(df, threshold=0.75)
    print(f"  After deduplication: {len(df)}")

    # ── output — no queue column ──────────────────────────────────────────
    kb_df = df[[
        'subject', 'query_clean', 'answer_clean',
        'category', 'priority',
        'tag_1_clean', 'tag_2_clean', 'tag_3_clean'
    ]].rename(columns={
        'query_clean' : 'query',
        'answer_clean': 'answer',
    })

    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    kb_df.to_csv(CSV_OUT, index=False)
    print(f"\nSaved cleaned CSV → {CSV_OUT}  ({len(kb_df)} rows)")

    print("\nCategory distribution in KB:")
    print(kb_df['category'].value_counts().to_string())

    total = len(kb_df)
    print(f"\nTag coverage:  "
          f"tag_1={(kb_df['tag_1_clean'] != '').sum()}/{total}  "
          f"tag_2={(kb_df['tag_2_clean'] != '').sum()}/{total}  "
          f"tag_3={(kb_df['tag_3_clean'] != '').sum()}/{total}")

    os.makedirs(os.path.dirname(JSONL_OUT), exist_ok=True)
    count = 0
    with open(JSONL_OUT, 'w') as f:
        for i, row in kb_df.iterrows():
            tags = [
                t for t in [row['tag_1_clean'], row['tag_2_clean'], row['tag_3_clean']]
                if t
            ]
            chunk = {
                "id"      : f"chunk_{i}",
                # embed subject + answer together for richer semantic matching
                "text"    : f"{row['subject']}. {row['answer']}".strip(". "),
                "answer"  : row['answer'],   # raw answer used by generate()
                "query"   : row['query'],
                "subject" : str(row['subject'] or ''),
                "category": row['category'],
                "priority": str(row['priority'] or ''),
                "tags"    : tags,
            }
            f.write(json.dumps(chunk) + '\n')
            count += 1

    print(f"\nSaved JSONL → {JSONL_OUT}  ({count} chunks)")
    print("\nSample chunk:")
    with open(JSONL_OUT) as f:
        print(json.dumps(json.loads(f.readline()), indent=2))


if __name__ == '__main__':
    prepare()