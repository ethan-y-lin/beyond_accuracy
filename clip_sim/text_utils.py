import re
from rapidfuzz import fuzz

# === Utility Functions ===
def normalize(text):
    return re.sub(r"[^\w\s]", "", text.lower()).strip()

def preprocess_text(df):
    df['text_lower'] = df['text'].str.lower()
    return df

def exact_match(concept, text):
    return concept in text

def fuzzy_match_factory(threshold=80):
    def fuzzy_match(concept, text):
        return fuzz.token_set_ratio(concept, text) >= threshold
    return fuzzy_match
