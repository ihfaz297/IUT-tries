import math
import re
import os
import json
import numpy as np
from collections import Counter
import pandas as pd

# ==========================================
# 0. OFFLINE CORPUS RETRIEVER (Mini-RAG)
# ==========================================

CORPUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "offline_corpus.json")

def _bn_words(text):
    """Tokenize Bengali text into words (Bengali unicode + ASCII tokens)."""
    return re.findall(r"[\u0980-\u09FF]+|\w+", str(text).lower())

class CorpusRetriever:
    """
    Lightweight TF-IDF retrieval engine.
    Loads offline_corpus.json once at init. For a given query,
    returns the best-matching paragraph and a similarity score.
    Zero external dependencies beyond numpy.
    """
    def __init__(self, corpus_path=CORPUS_PATH):
        self.paragraphs = []
        self.sources = []
        self.vocab = {}        # word -> index
        self.idf = None        # numpy array
        self.tfidf_matrix = None  # (n_docs, vocab_size)
        
        if not os.path.exists(corpus_path):
            print(f"[CorpusRetriever] WARNING: {corpus_path} not found. Retrieval features disabled.")
            self.enabled = False
            return
        
        with open(corpus_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.paragraphs = [d["text"] for d in data]
        self.sources = [d["source"] for d in data]
        
        if len(self.paragraphs) == 0:
            self.enabled = False
            return
        
        self.enabled = True
        self._build_index()
        print(f"[CorpusRetriever] Loaded {len(self.paragraphs)} paragraphs from corpus.")

    def _build_index(self):
        """Build TF-IDF vectors for all paragraphs."""
        # Step 1: Build vocabulary from all documents
        doc_word_counts = []
        doc_freq = Counter()  # how many docs contain each word
        
        for para in self.paragraphs:
            words = _bn_words(para)
            wc = Counter(words)
            doc_word_counts.append(wc)
            for w in set(words):
                doc_freq[w] += 1
        
        # Only keep words that appear in >= 2 docs (filters noise)
        vocab_words = [w for w, freq in doc_freq.items() if freq >= 2]
        self.vocab = {w: i for i, w in enumerate(vocab_words)}
        V = len(self.vocab)
        N = len(self.paragraphs)
        
        # Step 2: Compute IDF
        self.idf = np.zeros(V)
        for w, idx in self.vocab.items():
            self.idf[idx] = math.log((N + 1) / (doc_freq[w] + 1)) + 1  # smoothed IDF
        
        # Step 3: Build TF-IDF matrix
        self.tfidf_matrix = np.zeros((N, V))
        for doc_i, wc in enumerate(doc_word_counts):
            total = sum(wc.values())
            for w, c in wc.items():
                if w in self.vocab:
                    tf = c / total
                    self.tfidf_matrix[doc_i, self.vocab[w]] = tf * self.idf[self.vocab[w]]
        
        # Normalize rows for cosine similarity
        norms = np.linalg.norm(self.tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.tfidf_matrix = self.tfidf_matrix / norms

    def query(self, text, top_k=1):
        """
        Returns (best_score, best_paragraph, best_source) for the given text.
        Score is cosine similarity (0 to 1). Higher = better match.
        """
        if not self.enabled:
            return 0.0, "", ""
        
        words = _bn_words(text)
        if not words:
            return 0.0, "", ""
        
        # Build query vector
        wc = Counter(words)
        total = sum(wc.values())
        q_vec = np.zeros(len(self.vocab))
        for w, c in wc.items():
            if w in self.vocab:
                tf = c / total
                q_vec[self.vocab[w]] = tf * self.idf[self.vocab[w]]
        
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return 0.0, "", ""
        q_vec = q_vec / q_norm
        
        # Cosine similarity against all docs
        scores = self.tfidf_matrix @ q_vec
        best_idx = np.argmax(scores)
        return float(scores[best_idx]), self.paragraphs[best_idx], self.sources[best_idx]

    def score_grounding(self, prompt, response):
        """
        Composite grounding score: how well does the response match
        corpus evidence retrieved for the prompt?
        Returns a float 0-1. High = response is well-grounded.
        """
        if not self.enabled:
            return 0.0
        
        # Retrieve the best corpus paragraph for this prompt
        prompt_score, best_para, _ = self.query(prompt)
        
        if prompt_score < 0.05:
            # Prompt doesn't match anything in our corpus — can't judge
            return 0.0
        
        # Now check: does the response align with the retrieved evidence?
        resp_score, _, _ = self.query(response)
        
        # Also check overlap between response and the specific retrieved paragraph
        resp_words = set(_bn_words(response))
        para_words = set(_bn_words(best_para))
        if not resp_words:
            return 0.0
        overlap = len(resp_words & para_words) / len(resp_words)
        
        # Blend: retrieval relevance + direct word overlap
        return 0.5 * resp_score + 0.5 * overlap


# Singleton — loaded once when joggota_core is imported
_corpus_retriever = CorpusRetriever()


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
    
    # 3. Deterministic Joggota
    df_out['deterministic_joggota'] = df_out.apply(
        lambda row: deterministic_lexical_joggota(row['prompt_bn'], row['response_bn'], row['task_type']), axis=1
    )
    
    # 3b. Cultural Default Flag
    df_out['cultural_default_flag'] = df_out.apply(
        lambda row: cultural_default_penalty(row['prompt_bn'], row['response_bn']), axis=1
    )
    
    # 4. Corpus Retrieval Grounding (Mini-RAG)
    print("Computing corpus grounding scores...")
    df_out['corpus_match_score'] = df_out.apply(
        lambda row: _corpus_retriever.score_grounding(row['prompt_bn'], row['response_bn']), axis=1
    )
    
    return df_out
