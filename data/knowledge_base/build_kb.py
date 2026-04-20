# data/knowledge_base/build_kb.py

import os, sys, json
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
from src.preprocessing.text_cleaner import clean_body, CATEGORY_MERGE

DATA_PATH  = 'data/raw/cust_support_ticket.csv'
OUTPUT_PATH = 'data/knowledge_base/kb_chunks.jsonl'

def build():
    df = pd.read_csv(DATA_PATH)
    df = df[df['language'] == 'en'].copy()
    df['category'] = df['queue'].replace(CATEGORY_MERGE)
    df = df.dropna(subset=['answer'])

    chunks = []
    for i, row in df.iterrows():
        chunk = {
            "id"      : f"chunk_{i}",
            "text"    : str(row['answer']),          # what gets embedded + retrieved
            "query"   : str(row.get('body', '')),    # the original query this answers
            "subject" : str(row.get('subject', '')),
            "category": str(row['category']),
            "tags": list({
                tag.strip().lower()
                for j in range(1, 4)
                if pd.notna(row.get(f'tag_{j}'))
                for tag in str(row.get(f'tag_{j}', '')).split(',')
                if tag.strip()
            })
        }
        chunks.append(chunk)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')

    print(f"Saved {len(chunks)} chunks → {OUTPUT_PATH}")

if __name__ == '__main__':
    build()