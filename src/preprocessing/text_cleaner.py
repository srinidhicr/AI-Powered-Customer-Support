# src/preprocessing/text_cleaner.py

import re
import pandas as pd

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

LEAKAGE_KEYWORDS = [
    'billing', 'payment',
    'technical', 'tech', 'support',
    'sales', 'return', 'exchange',
    'hr', 'human resources', 'it',
    'customer service', 'customer support',
    'product support', 'technical support',
    'service', 'issue', 'request'
]

CATEGORY_MERGE = {
    'IT Support'                     : 'Technical',
    'Technical Support'              : 'Technical',
    'Service Outages and Maintenance': 'Technical',
    'Product Support'                : 'Product Inquiry',
    'Customer Service'               : 'Product Inquiry',
    'Sales and Pre-Sales'            : 'Product Inquiry',
}


def clean_body(text: str) -> str:
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = text.replace("\\n", " ").replace("\n", " ")
    for pattern in STRIP_PATTERNS:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)   # English only — no ä ö ü ß
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_tags(row, tag_cols=None) -> list:
    if tag_cols is None:
        tag_cols = ['tag_1', 'tag_2', 'tag_3']
    combined = []
    for col in tag_cols:
        if col in row and pd.notna(row[col]):
            parts = str(row[col]).split(',')
            combined.extend([p.strip().lower() for p in parts if p.strip()])
    seen, unique = set(), []
    for tag in combined:
        if tag not in seen:
            seen.add(tag)
            unique.append(tag)
    return unique[:3]


def remove_leakage(tags: list) -> list:
    return [
        tag for tag in tags
        if not any(kw in tag for kw in LEAKAGE_KEYWORDS)
    ]


def build_final_text(row) -> str:
    subject    = str(row.get('subject', '') or '')
    body_clean = str(row.get('body_clean', '') or '')
    tags_text  = str(row.get('tags_text', '') or '')
    return f"{subject} - {body_clean} {tags_text}".strip()


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Filter English immediately (as per Colab)
    df = df[df['language'] == 'en'].copy()

    # 2. Map Categories
    df['category'] = df['queue'].replace(CATEGORY_MERGE)

    # 3. Clean Body and Subject
    df['body_clean'] = df['body'].apply(clean_body)
    df['subject'] = df['subject'].fillna('')

    # 4. Create the 'text' column BEFORE filtering
    df['text'] = (df['subject'] + ' - ' + df['body_clean']).str.strip()
    df['text'] = df['text'].str.replace(r'^\s*-\s*', '', regex=True)

    # 5. Length Filter (This must match Colab row count)
    df['text_len'] = df['text'].str.split().str.len()
    df = df[df['text_len'] > 10].reset_index(drop=True)

    # 6. Tags (Normalized and Cleaned)
    df['tags_list'] = df.apply(normalize_tags, axis=1)
    df['tags_clean'] = df['tags_list'].apply(remove_leakage)
    df['tags_text'] = df['tags_clean'].apply(lambda x: ' '.join(x))

    # 7. Final Output String
    df['final_text'] = (df['text'] + ' ' + df['tags_text']).str.strip()

    # Drop too-short queries
    df['text_len'] = df['final_text'].str.split().str.len()
    df = df[df['text_len'] > 5].reset_index(drop=True)

    return df