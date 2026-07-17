"""
Ground-truth-source matcher (see CLAUDE.md "Ground-Truth Source Expansion").

One reusable matcher, not N one-off ones: given a competition (prompt_bn, response_bn),
find the closest question in a pool of real QA datasets (NCTB textbooks, TyDi QA gold
passages) and, if the match is close enough to trust, compare response_bn against the
dataset's own answer. Conservative by design -- near-exact matching only (char-ngram
TF-IDF cosine above a high threshold), not fuzzy semantic search. A false-positive match
injects *wrong* ground truth, which is worse than no signal at all.

Sources wired up:
  - NCTB-QA (ShihabReza/nctb-qa on HF, ungated) -- 5,812 Bengali QA pairs built from real
    NCTB Class 6-9 ICT/Science textbooks. Most aligned with the competition's stated Phase 2
    source domains (NCTB textbooks are named explicitly in some_catches.txt).
  - TyDi QA Bengali gold-passage (local, curious_insight/) -- 2,503 general-knowledge QA
    pairs, zero download cost.

BnMMLU (Alvee's real +0.038 lever) is NOT wired up here -- its HuggingFace dataset repo is
gated ("manual" approval required), confirmed via a direct 401 on the parquet endpoint. Only
GitHub code/scripts are public, no data. Needs either HF access-request approval or Alvee's
own copy/token before it can be added the same way.

Output: two soft features, not a hard override (deliberate departure from Alvee's own
math_override, which fires on 0/299 training rows and directly overrides test rows blind):
  - gt_match_score   -- best cosine similarity to any indexed question (0 if nothing close)
  - gt_agreement     -- token-overlap agreement between response_bn and the matched answer,
                        only computed when gt_match_score clears MATCH_THRESHOLD; 0.5
                        (neutral) otherwise so unmatched rows don't bias a downstream model
                        toward either class.
"""
import os
import re
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MATCH_THRESHOLD = 0.75  # conservative: near-exact question match only, per the integration plan

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[।!?,;:\"'\(\)\[\]]+")
_BN2EN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_BN_NUMBER_WORDS = {
    "শূন্য": 0, "এক": 1, "দুই": 2, "তিন": 3, "চার": 4, "পাঁচ": 5, "ছয়": 6, "সাত": 7,
    "আট": 8, "নয়": 9, "দশ": 10, "এগারো": 11, "বারো": 12, "তেরো": 13, "চৌদ্দ": 14,
    "পনেরো": 15, "ষোলো": 16, "সতেরো": 17, "আঠারো": 18, "উনিশ": 19, "বিশ": 20,
    "ত্রিশ": 30, "চল্লিশ": 40, "পঞ্চাশ": 50, "ষাট": 60, "সত্তর": 70, "আশি": 80,
    "নব্বই": 90, "শত": 100, "হাজার": 1000, "লক্ষ": 100000, "কোটি": 10000000,
}


def _extract_numbers(text):
    """Digits (Bengali or Latin) plus a best-effort read of spelled-out Bengali
    number words (e.g. "আঠারশ বত্রিশ" -> 1832), reusing the same word list joggota_core.py
    uses for its math rule. Not a full parser -- good enough to stop numeric answers from
    getting silently downgraded to partial token-overlap credit."""
    if not isinstance(text, str):
        return set()
    latin = text.translate(_BN2EN_DIGITS)
    nums = {m for m in re.findall(r"\d+", latin)}
    total, hit = 0, False
    for tok in bn_tokenize(text):
        if tok in _BN_NUMBER_WORDS:
            v = _BN_NUMBER_WORDS[tok]
            if v >= 100:
                total = (total or 1) * v
            else:
                total += v
            hit = True
    if hit and total:
        nums.add(str(total))
    return nums


def normalize(text):
    if not isinstance(text, str):
        return ""
    t = _PUNCT_RE.sub(" ", text)
    t = _WS_RE.sub(" ", t).strip()
    return t


def bn_tokenize(text):
    return re.findall(r"[ঀ-৿]+|\S+", normalize(text))


NCTB_QA_URL = "https://huggingface.co/datasets/ShihabReza/nctb-qa/resolve/main/data/train-00000-of-00001.parquet"


def load_nctb_qa(path="nctb_qa_train.parquet"):
    if not os.path.exists(path):
        # Ungated, public dataset (confirmed via a direct 200 on this resolve URL) --
        # safe to fetch at runtime. Phase 1 permits internet; Phase 2 needs this file
        # pre-downloaded and bundled instead (see CLAUDE.md Key Constraints).
        try:
            import urllib.request
            print(f"  {path} not found locally -- downloading from HuggingFace...")
            urllib.request.urlretrieve(NCTB_QA_URL, path)
        except Exception as e:
            print(f"  NCTB-QA download failed ({e}) -- continuing without this source")
            return pd.DataFrame(columns=["question", "answer", "source_name"])
    df = pd.read_parquet(path)
    df = df[df["language"] == "bn"][["question", "answer"]].dropna()
    df["source_name"] = "nctb_qa"
    return df.reset_index(drop=True)


TYDIQA_URLS = [
    "https://huggingface.co/datasets/google-research-datasets/tydiqa/resolve/main/secondary_task/train-00000-of-00001.parquet",
    "https://huggingface.co/datasets/google-research-datasets/tydiqa/resolve/main/secondary_task/validation-00000-of-00001.parquet",
]


def load_tydiqa(path="curious_insight/Sample Selection for QA/Datasets/tydiqa_goldp_bengali.csv",
                 cache_path="tydiqa_goldp_bengali.parquet"):
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[["question", "answer_text"]].dropna().rename(columns={"answer_text": "answer"})
        df["source_name"] = "tydiqa"
        return df.reset_index(drop=True)

    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        df["source_name"] = "tydiqa"
        return df.reset_index(drop=True)

    # Neither the original local CSV nor a prior download cache exists -- fetch straight
    # from the upstream TyDi QA "secondary_task" (gold-passage) config, ungated, confirmed
    # via a direct 200. This is the exact source the local CSV was itself built from: train
    # (2,390 Bengali rows) + validation (113 Bengali rows) = 2,503, matching the local CSV's
    # row count exactly. Downloads the full multilingual file (~27MB, all languages) since
    # HF doesn't expose a per-language split for this config, then filters to Bengali only
    # and caches the filtered result so this only happens once.
    try:
        import urllib.request
        print(f"  {path} not found locally -- downloading TyDi QA secondary_task from HuggingFace...")
        parts = []
        for i, url in enumerate(TYDIQA_URLS):
            tmp = f"_tydiqa_tmp_{i}.parquet"
            urllib.request.urlretrieve(url, tmp)
            raw = pd.read_parquet(tmp)
            parts.append(raw[raw["id"].str.startswith("bengali")])
            os.remove(tmp)
        bn = pd.concat(parts, ignore_index=True)
        bn["answer"] = bn["answers"].map(
            lambda a: a["text"][0] if isinstance(a, dict) and len(a.get("text", [])) else None
        )
        df = bn[["question", "answer"]].dropna().reset_index(drop=True)
        df.to_parquet(cache_path)
    except Exception as e:
        print(f"  TyDi QA download failed ({e}) -- continuing without this source")
        return pd.DataFrame(columns=["question", "answer", "source_name"])

    df["source_name"] = "tydiqa"
    return df.reset_index(drop=True)


INDICQA_BN_URL = "https://huggingface.co/datasets/ai4bharat/IndicQA/resolve/main/data/indicqa.bn.json"


def load_indicqa(path="indicqa_bn.json"):
    """IndicQA Bengali (ai4bharat/IndicQA, CC-BY-4.0, ungated -- confirmed 200).
    SQuAD-format nesting: data[].paragraphs[].qas[] with `question` + `answers[].text`."""
    if not os.path.exists(path):
        try:
            import urllib.request
            print(f"  {path} not found locally -- downloading IndicQA-bn from HuggingFace...")
            urllib.request.urlretrieve(INDICQA_BN_URL, path)
        except Exception as e:
            print(f"  IndicQA download failed ({e}) -- continuing without this source")
            return pd.DataFrame(columns=["question", "answer", "source_name"])
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    rows = []
    for art in d.get("data", []):
        for para in art.get("paragraphs", []):
            for qa in para.get("qas", []):
                answers = qa.get("answers") or []
                if not answers:
                    continue
                text = str(answers[0].get("text", "")).strip()
                if text:
                    rows.append({"question": qa.get("question", ""), "answer": text})
    df = pd.DataFrame(rows)
    if len(df):
        df["source_name"] = "indicqa"
    return df


BANGLARQA_URL = "https://huggingface.co/datasets/sartajekram/BanglaRQA/resolve/main/Train.json"


def load_banglarqa(path="banglarqa_train.json"):
    """BanglaRQA (sartajekram/BanglaRQA, ungated -- confirmed 200), Bengali Wikipedia passages.
    Nesting: data[].qas[] with `question_text`, `is_answerable`, `question_type`,
    `answers.answer_text[]`.

    Two filters matter here. Unanswerable questions carry empty answers and must be dropped.
    Yes/no ("confirmation") questions are dropped too: an answer of "হ্যাঁ"/"না" cannot
    discriminate a faithful response from a hallucinated one, and would fire agreement=1.0
    on any response that happens to contain the token."""
    if not os.path.exists(path):
        try:
            import urllib.request
            print(f"  {path} not found locally -- downloading BanglaRQA from HuggingFace...")
            urllib.request.urlretrieve(BANGLARQA_URL, path)
        except Exception as e:
            print(f"  BanglaRQA download failed ({e}) -- continuing without this source")
            return pd.DataFrame(columns=["question", "answer", "source_name"])
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    rows = []
    for rec in d.get("data", []):
        for qa in rec.get("qas", []):
            if str(qa.get("is_answerable", "0")) != "1":
                continue
            if qa.get("question_type") == "confirmation":
                continue
            texts = (qa.get("answers") or {}).get("answer_text") or []
            text = next((str(t).strip() for t in texts if str(t).strip()), "")
            if text:
                rows.append({"question": qa.get("question_text", ""), "answer": text})
    df = pd.DataFrame(rows)
    if len(df):
        df["source_name"] = "banglarqa"
    return df


BANGLA_MMLU_URLS = [
    "https://huggingface.co/datasets/hishab/bangla-mmlu/resolve/main/data/validation-00000-of-00001.parquet",
    "https://huggingface.co/datasets/hishab/bangla-mmlu/resolve/main/data/test-00000-of-00001.parquet",
]
_MCQ_LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def load_bangla_mmlu(cache_path="bangla_mmlu.parquet"):
    """hishab/bangla-mmlu -- ~88K Bengali exam MCQs (ungated, confirmed 200).

    This is the single most important source in the pool, and it is here for a measured
    reason: the Wikipedia-derived sources above match 92% of *context* rows but only 1.8%
    of *no-context* rows, while this one is almost exactly the mirror image -- 38.9% of
    no-context rows, 0.8% of context rows. No-context rows are the pipeline's dead segment
    (~0.50 hallu-F1, chance), so this source is the only one aimed at it. It stands in for
    the gated samanjoy2/BnMMLU (401, manual approval) that a teammate used for a real
    +0.038 leaderboard gain -- same exam-bank shape, freely accessible.

    Schema: question + choices[] + answer as a letter, so the letter is resolved back into
    the actual option text before use.
    """
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        df["source_name"] = "bangla_mmlu"
        return df.reset_index(drop=True)

    try:
        import urllib.request
        print(f"  {cache_path} not found locally -- downloading bangla-mmlu from HuggingFace...")
        parts = []
        for i, url in enumerate(BANGLA_MMLU_URLS):
            tmp = f"_bmmlu_tmp_{i}.parquet"
            urllib.request.urlretrieve(url, tmp)
            parts.append(pd.read_parquet(tmp))
            os.remove(tmp)
        raw = pd.concat(parts, ignore_index=True)

        def _resolve(row):
            choices = list(row["choices"]) if row["choices"] is not None else []
            idx = _MCQ_LETTER_TO_IDX.get(str(row["answer"]).strip().upper())
            if idx is None or idx >= len(choices):
                return None
            return choices[idx]

        raw["answer_text"] = raw.apply(_resolve, axis=1)
        df = raw[["question", "answer_text"]].dropna().rename(columns={"answer_text": "answer"})
        df = df[df["question"].astype(str).str.strip().astype(bool)].reset_index(drop=True)
        df.to_parquet(cache_path)
    except Exception as e:
        print(f"  bangla-mmlu download failed ({e}) -- continuing without this source")
        return pd.DataFrame(columns=["question", "answer", "source_name"])

    df["source_name"] = "bangla_mmlu"
    return df.reset_index(drop=True)


def load_all_sources():
    parts = [load_nctb_qa(), load_tydiqa(), load_indicqa(), load_banglarqa(), load_bangla_mmlu()]
    parts = [p for p in parts if len(p)]
    if not parts:
        raise RuntimeError("no ground-truth QA sources found -- check paths")
    pool = pd.concat(parts, ignore_index=True)
    pool = pool[pool["question"].map(lambda s: len(normalize(s)) > 0)].reset_index(drop=True)
    return pool


class GroundTruthMatcher:
    def __init__(self, pool=None):
        self.pool = pool if pool is not None else load_all_sources()
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
        self.index = self.vec.fit_transform(self.pool["question"].map(normalize))
        print(f"GroundTruthMatcher: indexed {len(self.pool)} QA pairs "
              f"({self.pool['source_name'].value_counts().to_dict()})")

    def _best_match(self, prompts, batch=256):
        """Returns (best_idx, best_score) per prompt, batched to bound memory."""
        n = len(prompts)
        best_idx = np.full(n, -1, dtype=int)
        best_score = np.zeros(n)
        qv = self.vec.transform([normalize(p) for p in prompts])
        for i in range(0, n, batch):
            sims = cosine_similarity(qv[i:i + batch], self.index)
            idx = sims.argmax(axis=1)
            score = sims[np.arange(sims.shape[0]), idx]
            best_idx[i:i + batch] = idx
            best_score[i:i + batch] = score
        return best_idx, best_score

    @staticmethod
    def _agreement(response, answer):
        """Numeric answers first (exact digit match required -- no partial credit for a
        wrong number that happens to share surrounding words, e.g. "৭২০ মিটার" vs the
        correct "৬৭০ মিটার"), then token-overlap agreement: containment (short exam
        answers are often substrings/superstrings of the response), Jaccard as fallback."""
        r_norm, a_norm = normalize(response), normalize(answer)
        if not r_norm or not a_norm:
            return 0.5
        a_nums = _extract_numbers(answer)
        if a_nums:
            r_nums = _extract_numbers(response)
            return 1.0 if (a_nums & r_nums) else 0.0
        if a_norm in r_norm or r_norm in a_norm:
            return 1.0
        rt, at = set(bn_tokenize(response)), set(bn_tokenize(answer))
        if not rt or not at:
            return 0.5
        overlap = len(rt & at) / len(at)  # recall against the known answer's tokens
        return float(overlap)

    def score(self, prompts, responses):
        idx, sim = self._best_match(prompts)
        n = len(prompts)
        match_score = sim.copy()
        agreement = np.full(n, 0.5)
        matched = sim >= MATCH_THRESHOLD
        answers = self.pool["answer"].values
        for i in np.where(matched)[0]:
            agreement[i] = self._agreement(responses[i], answers[idx[i]])
        return match_score, agreement, matched, idx


def compute_gt_features(df, matcher=None):
    matcher = matcher or GroundTruthMatcher()
    prompts = df["prompt_bn"].tolist()
    responses = df["response_bn"].tolist()
    match_score, agreement, matched, idx = matcher.score(prompts, responses)
    out = df.copy()
    out["gt_match_score"] = match_score
    out["gt_agreement"] = agreement
    out["gt_matched"] = matched.astype(int)
    return out, idx


if __name__ == "__main__":
    import json

    with open("dataset samples.json", encoding="utf-8") as f:
        train_raw = pd.DataFrame(json.load(f))

    matcher = GroundTruthMatcher()
    feat, idx = compute_gt_features(train_raw, matcher)

    n_matched = int(feat["gt_matched"].sum())
    print(f"\nMatched {n_matched}/{len(feat)} training rows at threshold {MATCH_THRESHOLD}")

    matched_rows = feat[feat["gt_matched"] == 1]
    if len(matched_rows):
        # does agreement correlate with the true label at all? (label=1 faithful should
        # skew toward high agreement, label=0 hallucinated toward low agreement)
        corr = matched_rows[["gt_agreement"]].assign(label=matched_rows["label"]).corr().iloc[0, 1]
        print(f"corr(gt_agreement, label) on matched rows: {corr:.3f}")
        print(matched_rows.groupby("label")["gt_agreement"].agg(["mean", "count"]))

        print("\n--- manual spot-check sample (first 15 matches) ---")
        for i, (_, row) in enumerate(matched_rows.head(15).iterrows()):
            j = idx[row.name]
            print(f"\n[{i}] sim={row['gt_match_score']:.3f} agreement={row['gt_agreement']:.2f} true_label={row['label']}")
            print("  prompt:  ", row["prompt_bn"][:100])
            print("  matched Q:", matcher.pool['question'].iat[j][:100])
            print("  response:", row["response_bn"][:80])
            print("  gt answer:", str(matcher.pool['answer'].iat[j])[:80])
    else:
        print("No matches at all -- threshold may be too strict, or prompts don't overlap the sources.")
