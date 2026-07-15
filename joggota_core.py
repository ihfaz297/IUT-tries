import math
import re
import numpy as np
from collections import Counter
import pandas as pd

def _bn_words(text):
    """Tokenize Bengali text into words (Bengali unicode + ASCII tokens)."""
    return re.findall(r"[\u0980-\u09FF]+|\w+", str(text).lower())

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
# 1b. CONTEXT GROUNDING & RESPONSE QUALITY
# ==========================================

def _ngrams(words, n):
    return set(zip(*[words[i:] for i in range(n)])) if len(words) >= n else set()

def context_containment(context, response):
    """
    Fraction of the response's word bigrams that appear verbatim in the context.
    High score = response text is directly lifted from context (strong faithfulness signal).
    Only meaningful for context rows.
    """
    if pd.isna(context) or str(context).strip() in {"", "[NULL]", "nan", "NaN"}:
        return 0.0
    ctx_words = _bn_words(context)
    resp_words = _bn_words(response)
    resp_bigrams = _ngrams(resp_words, 2)
    if not resp_bigrams:
        return 0.0
    ctx_bigrams = _ngrams(ctx_words, 2)
    return len(resp_bigrams & ctx_bigrams) / len(resp_bigrams)

REFUSAL_PHRASES = [
    "আমি জানি না", "আমার জানা নেই", "দুঃখিত", "ক্ষমা করবেন", "ক্ষমাপ্রার্থী",
    "উত্তর দেওয়া সম্ভব নয়", "নিশ্চিত নই", "বলতে পারছি না", "তথ্য নেই",
]

def resp_is_refusal(response):
    """Flags deflection/refusal responses instead of an actual answer."""
    r = str(response)
    return 1.0 if any(phrase in r for phrase in REFUSAL_PHRASES) else 0.0

def resp_code_switch_ratio(response):
    """Ratio of Latin-alphabet word tokens to total word tokens in the response."""
    words = _bn_words(response)
    if not words:
        return 0.0
    latin = sum(1 for w in words if re.fullmatch(r"[a-z0-9]+", w))
    return latin / len(words)

def resp_repetition_score(response):
    """1 - (unique bigrams / total bigrams). Higher = more internal repetition."""
    words = _bn_words(response)
    if len(words) < 2:
        return 0.0
    bigrams = list(zip(words, words[1:]))
    if not bigrams:
        return 0.0
    return 1.0 - (len(set(bigrams)) / len(bigrams))

def resp_is_question(response):
    """Response deflects by ending with a question mark instead of answering."""
    r = str(response).strip()
    return 1.0 if r.endswith("?") else 0.0


# ==========================================
# 2. DETERMINISTIC JOGGOTA (Rules & Regex)
# ==========================================

_MATH_KEYWORDS = ["সম্ভাবনা", "যোগ", "বিয়োগ", "গুণ", "ভাগ", "সংখ্যা",
                  "সমীকরণ", "ক্ষেত্রফল", "পরিসীমা", "লসাগু", "গসাগু"]

_BN_NUMBER_WORDS = [
    "শূন্য", "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়", "দশ",
    "এগার", "বার", "তের", "চৌদ্দ", "পনের", "ষোল", "সতের", "আঠার", "উনিশ", "বিশ",
    "ত্রিশ", "চল্লিশ", "পঞ্চাশ", "ষাট", "সত্তর", "আশি", "নব্বই",
    "শত", "হাজার", "লক্ষ", "কোটি",
]
_MCQ_OPTIONS = {"ক", "খ", "গ", "ঘ", "ঙ"}

def classify_task(prompt: str) -> str:
    """Classifies the task type to route the validation logic."""
    p = str(prompt).lower()
    if re.search(r'বাগধারা|প্রবাদ|প্রবচন', p): return "idiom"
    if re.search(r'অর্থ|ভাবার্থ|শাব্দিক|সমার্থক|বিপরীত|প্রতিশব্দ', p): return "vocabulary"
    if re.search(r'বানান|শুদ্ধ বানান', p): return "spelling"
    # Prefix match on whitespace tokens, not substring search on the raw string --
    # "ভাগ" (divide) and "যোগ" (add) are short enough to false-positive as substrings
    # inside unrelated words like "বিভাগ" (department) and "প্রতিযোগিতা" (competition).
    # Bengali suffixes attach at the end of a stem, so a real match always starts the token.
    tokens = p.split()
    if any(tok.startswith(kw) for tok in tokens for kw in _MATH_KEYWORDS):
        return "math"
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
        # \d matches Bengali digits (০-৯) too, but misses spelled-out number words
        # ("সাতটি" = "seven") and MCQ letter answers ("গ" = option C) -- both are
        # valid, non-hallucinated responses that don't contain a literal digit.
        resp_numbers = re.findall(r'\d+', resp_clean)
        has_number_word = any(w in resp_clean for w in _BN_NUMBER_WORDS)
        is_mcq_letter = resp_clean.strip('।.?!০-৯0-9') in _MCQ_OPTIONS
        if not resp_numbers and not has_number_word and not is_mcq_letter:
            return 0.0
            
    if task_type == "idiom":
        # Idioms require figurative meaning, not literal.
        COMMON_IDIOMS = {
            "অকাল কুষ্মাণ্ড": {"literal": ["কুমড়া", "সবজি", "ফল"], "figurative": ["অপদার্থ", "অযোগ্য", "বাজে", "অকাজের"]},
            "আকাশ কুসুম": {"literal": ["আকাশ", "ফুল"], "figurative": ["অসম্ভব", "কল্পনা", "অবাস্তব", "মিথ্যা"]},
            "ঘোড়ার ডিম": {"literal": ["ঘোড়া", "ডিম", "অশ্ব"], "figurative": ["অসম্ভব", "অবাস্তব", "কিছুই না", "অস্তিত্বহীন"]},
            "গাছে কাঁঠাল গোঁফে তেল": {"literal": ["গাছ", "কাঁঠাল", "গোঁফ", "তেল"], "figurative": ["আগে", "প্রস্তুতি", "পাওয়ার", "আশায়"]},
            "চোখে সর্ষে ফুল দেখা": {"literal": ["সর্ষে", "ফুল", "চোখ"], "figurative": ["বিপদ", "দিশেহারা", "ঘোরগ্রস্ত", "অন্ধকার"]},
            "হাতের পাঁচ": {"literal": ["হাত", "পাঁচ", "আঙুল"], "figurative": ["সম্বল", "উপায়", "শেষ", "একমাত্র"]},
            "অন্ধের যষ্টি": {"literal": ["অন্ধ", "লাঠি", "যষ্টি"], "figurative": ["অবলম্বন", "একমাত্র", "ভরসা"]},
            "আদা জল খেয়ে লাগা": {"literal": ["আদা", "জল", "পানি"], "figurative": ["উৎসাহে", "চেষ্টা", "প্রাণপণ", "উঠেপড়ে"]},
            "আঙ্গুল ফুলে কলাগাছ": {"literal": ["আঙ্গুল", "কলাগাছ", "গাছ"], "figurative": ["বড়লোক", "উন্নতি", "ধনী", "হঠাৎ"]}
        }
        
        for idiom, keywords in COMMON_IDIOMS.items():
            if idiom in prompt_clean:
                # Check if the LLM took it literally
                has_literal = any(word in resp_clean for word in keywords["literal"])
                has_figurative = any(word in resp_clean for word in keywords["figurative"])
                
                if has_literal and not has_figurative:
                    return 0.0 # Absolute hallucination
                elif has_figurative:
                    return 1.0 # Passed deterministic check
            
    # For vocabulary, we would ideally hook in the local dictionary check here.
    # For now, we abstain if no strict rule was triggered.
    return -1.0 

def cultural_default_penalty(prompt: str, response: str) -> float:
    """
    Detects C1 band 'Cultural Default' hallucinations.
    Returns 1.0 if a known cultural default is detected in the response instead of the expected Bangladesh-specific truth, otherwise 0.0.
    """
    p = str(prompt).lower()
    r = str(response).lower()
    
    # 1. Yunus Nobel (Expected: Peace/শান্তি, Default: Economics/অর্থনীতি)
    if 'ইউনূস' in p or 'yunus' in p:
        if 'নোবেল' in p or 'nobel' in p:
            if 'অর্থনীতি' in r or 'economics' in r:
                return 1.0
                
    # 2. National Poet / Specific Authors (Expected: Nazrul/নজরুল or others, Default: Tagore/রবীন্দ্রনাথ)
    if 'কবি' in p or 'লেখক' in p or 'উপন্যাস' in p or 'কবিতা' in p:
        # If the prompt does NOT mention Tagore, but response DOES
        if 'রবীন্দ্রনাথ' not in p and 'রবীন্দ্রনাথ' in r:
            return 1.0
            
    # 3. ORS / Saline (Expected: Bangladesh/ICDDR,B/Rafiqul, Default: Western/পাশ্চাত্য/WHO)
    if 'স্যালাইন' in p or 'ors' in p or 'saline' in p:
        if 'পাশ্চাত্য' in r or 'western' in r or 'america' in r or 'মার্কিন' in r:
            return 1.0
            
    # 4. First President/PM (Expected: Mujib/Tajuddin, Default: Zia/Others depending on context)
    # This is trickier, keeping it simple for now.
    
    return 0.0

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

    # 2b. Context Grounding & Response Quality
    df_out['context_containment'] = df_out.apply(lambda row: context_containment(row['context'], row['response_bn']), axis=1)
    df_out['resp_is_refusal'] = df_out['response_bn'].apply(resp_is_refusal)
    df_out['resp_code_switch_ratio'] = df_out['response_bn'].apply(resp_code_switch_ratio)
    df_out['resp_repetition_score'] = df_out['response_bn'].apply(resp_repetition_score)
    df_out['resp_is_question'] = df_out['response_bn'].apply(resp_is_question)

    # 3. Deterministic Joggota
    df_out['deterministic_joggota'] = df_out.apply(
        lambda row: deterministic_lexical_joggota(row['prompt_bn'], row['response_bn'], row['task_type']), axis=1
    )
    
    # 3b. Cultural Default Flag
    df_out['cultural_default_flag'] = df_out.apply(
        lambda row: cultural_default_penalty(row['prompt_bn'], row['response_bn']), axis=1
    )

    return df_out
