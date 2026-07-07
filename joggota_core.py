import math
import re
from collections import Counter
import pandas as pd

# ==========================================
# 1. THE FORM ENGINE (Akangkha & Asotti)
# ==========================================

def word_entropy(text):
    """Measures structural randomness. High entropy often correlates with verbose hallucinations."""
    words = str(text).split()
    if len(words) < 2: return 0.0
    counts = Counter(words)
    total = len(words)
    return -sum((c/total) * math.log2(c/total) for c in counts.values())

def char_entropy(text):
    """Character-level entropy."""
    text = str(text)
    if len(text) < 2: return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c/total) * math.log2(c/total) for c in counts.values())

def novel_char_ratio(context, response):
    """Detects extrinsic hallucinations by finding characters in response not in context."""
    if pd.isna(context) or str(context).strip() in {"", "[NULL]", "nan", "NaN"}:
        return 0.0 # Only applies to RAG/Context rows
    
    ctx_chars  = set(str(context))
    resp_chars = set(str(response))
    return len(resp_chars - ctx_chars) / max(len(resp_chars), 1)

def response_length_ratio(prompt, response):
    """Hallucinated responses are on average 69% longer in this dataset."""
    p_len = len(str(prompt).strip())
    r_len = len(str(response).strip())
    return r_len / max(p_len, 1)


# ==========================================
# 2. DETERMINISTIC JOGGOTA (Rules & Regex)
# ==========================================

def classify_task(prompt: str) -> str:
    """Classifies the task type to route the validation logic."""
    p = str(prompt).lower()
    if re.search(r'বাগধারা|প্রবাদ|প্রবচন', p): return "idiom"
    if re.search(r'অর্থ|ভাবার্থ|শাব্দিক|সমার্থক|বিপরীত|প্রতিশব্দ', p): return "vocabulary"
    if re.search(r'বানান|শুদ্ধ বানান', p): return "spelling"
    if re.search(r'সম্ভাবনা|যোগ|বিয়োগ|গুণ|ভাগ|সংখ্যা|সমীকরণ|ক্ষেত্রফল|পরিসীমা|লসাগু|গসাগু', p): return "math"
    if re.search(r'অনুবাদ|ইংরেজি|translate', p): return "translation"
    if re.search(r'সমাস|ব্যাকরণ|কারক|বিভক্তি|ধাতু|প্রত্যয়|উপসর্গ', p): return "grammar"
    return "factual"

def deterministic_lexical_joggota(prompt: str, response: str, task_type: str) -> float:
    """
    Applies strict rule-based Joggota for specific task types.
    Returns:
      1.0 (Looks Faithful/Passed rule)
      0.0 (Definite Hallucination)
      -1.0 (Abstain - Rule doesn't apply)
    """
    prompt_clean = str(prompt).strip()
    resp_clean = str(response).strip()
    
    if task_type == "spelling":
        # If it's a spelling task, the response should ideally just be the correct word.
        # If the response is extremely long, it's likely hallucinating a whole explanation.
        if len(resp_clean.split()) > 5:
            return 0.0
            
    if task_type == "math":
        # If math, extract numbers. If there are NO numbers in response, it's likely a hallucinated refusal.
        resp_numbers = re.findall(r'\d+', resp_clean)
        if not resp_numbers:
            return 0.0
            
    if task_type == "idiom":
        # Idioms require figurative meaning, not literal.
        # TODO: Hook in a lookup dictionary to penalize literal translations.
        pass
            
    # For vocabulary, we would ideally hook in the local dictionary check here.
    # For now, we abstain if no strict rule was triggered.
    return -1.0 

def extract_joggota_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies the Form Engine and Deterministic Joggota to the entire dataframe."""
    df_out = df.copy()
    
    # Clean NaNs
    df_out['context'] = df_out['context'].fillna('[NULL]')
    df_out['prompt_bn'] = df_out['prompt_bn'].astype(str)
    df_out['response_bn'] = df_out['response_bn'].astype(str)
    
    # 1. Task Classification
    df_out['task_type'] = df_out['prompt_bn'].apply(classify_task)
    
    # 2. Form Engine Features
    df_out['word_entropy'] = df_out['response_bn'].apply(word_entropy)
    df_out['char_entropy'] = df_out['response_bn'].apply(char_entropy)
    df_out['novel_char_ratio'] = df_out.apply(lambda row: novel_char_ratio(row['context'], row['response_bn']), axis=1)
    df_out['length_ratio'] = df_out.apply(lambda row: response_length_ratio(row['prompt_bn'], row['response_bn']), axis=1)
    df_out['resp_len'] = df_out['response_bn'].str.len()
    
    # 3. Deterministic Joggota
    df_out['deterministic_joggota'] = df_out.apply(
        lambda row: deterministic_lexical_joggota(row['prompt_bn'], row['response_bn'], row['task_type']), axis=1
    )
    
    return df_out
