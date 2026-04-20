# scripts/diagnose_kb.py
import pandas as pd

df = pd.read_csv('data/processed/kb_clean.csv')

acknowledgement_patterns = [
    'please provide', 'could you please', 'kindly provide',
    'thank you for reaching out', 'we are working',
    'our team is investigating', 'we will look into',
    'to assist you further', 'to help us assist',
    'please describe', 'please specify', 'please confirm',
    'please share', 'please inform', 'please let'
]

def is_acknowledgement(text):
    t = str(text).lower()
    return sum(1 for p in acknowledgement_patterns if p in t) >= 2

df['is_ack'] = df['answer'].apply(is_acknowledgement)
print(f"Acknowledgement-only answers: {df['is_ack'].mean():.1%}")
print(f"Total KB entries: {len(df)}")
print(f"Potentially useful: {(~df['is_ack']).sum()}")
print("\nBy category:")
print(df.groupby('category')['is_ack'].mean().sort_values(ascending=False).to_string())