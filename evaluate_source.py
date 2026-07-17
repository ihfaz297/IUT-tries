"""
Is a candidate ground-truth source worth adding? Answers that in ~30 seconds, CPU-only.

WHY THIS EXISTS (read before using it):
The 0.636 -> 0.843 jump came from exactly one insight, and it is easy to miss:
**source SHAPE matters, source SIZE does not.**

  - Adding IndicQA + BanglaRQA more than DOUBLED the index (8,315 -> 17,568 QA pairs) and
    changed coverage by literally zero rows (125 -> 125 matched). Wasted effort.
  - Adding hishab/bangla-mmlu (exam MCQs) took coverage 125 -> 210 and the real leaderboard
    score 0.724 -> 0.843.

The difference: the Wikipedia-derived sources matched 92% of *context* rows (already saturated)
and 1.8% of *no-context* rows. The exam bank was the mirror image -- 38.9% of no-context rows.
No-context rows were the dead segment (~0.50 hallu-F1, chance). Only a differently-SHAPED
source could reach them.

So the question is never "is this dataset big?" It is:
    "which rows are still dead, and is this source shaped like THEM?"

This script answers that. For any candidate source it reports the metric that actually
predicts a leaderboard gain: **incremental coverage on the rows we currently miss**, split by
context / no-context, plus whether the agreement signal still correlates with the true label.

USAGE
-----
    from evaluate_source import evaluate_candidate
    import pandas as pd

    cand = pd.DataFrame({"question": [...], "answer": [...]})   # that's all it needs
    evaluate_candidate(cand, name="my-new-source")

Or run directly for a worked example against the current pool:
    python evaluate_source.py
"""
import json
import numpy as np
import pandas as pd

from ground_truth_matcher import (
    GroundTruthMatcher,
    compute_gt_features,
    load_all_sources,
    MATCH_THRESHOLD,
)

TRAIN_PATH = "dataset samples.json"


def _load_train():
    with open(TRAIN_PATH, encoding="utf-8") as f:
        df = pd.DataFrame(json.load(f))
    df["has_ctx"] = (
        ~df["context"].astype(str).str.strip().str.lower().isin(["[null]", "", "nan"])
    ).astype(int)
    return df


def _coverage_report(feat, train, label):
    """Coverage + correlation, split by context/no-context -- the split is the whole point.
    An aggregate coverage number hides exactly the thing that matters."""
    out = {}
    for ctx, name in [(1, "context"), (0, "no_context")]:
        sub = feat[train["has_ctx"].values == ctx]
        matched = sub[sub["gt_matched"] == 1]
        corr = np.nan
        if len(matched) > 3 and matched["label"].nunique() > 1:
            corr = matched[["gt_agreement"]].assign(label=matched["label"]).corr().iloc[0, 1]
        out[name] = {
            "rows": len(sub),
            "matched": len(matched),
            "rate": len(matched) / max(len(sub), 1),
            "corr": corr,
        }
    out["total_matched"] = int(feat["gt_matched"].sum())
    return out


def evaluate_candidate(candidate, name="candidate", baseline_pool=None, verbose=True):
    """
    candidate: DataFrame with columns `question` and `answer` (answer already resolved to
               its actual text -- if the source stores an MCQ letter, resolve it first, see
               load_bangla_mmlu's _resolve for how).

    Reports three things, in increasing order of importance:
      1. the candidate ALONE -- does it match anything at all?
      2. the candidate's split -- context vs no-context. This is the shape test.
      3. the INCREMENTAL gain when added to the existing pool. This is the only number that
         predicts a real score change. A source can look great alone and add nothing on top
         of what we already have (IndicQA/BanglaRQA did exactly this).
    """
    train = _load_train()
    candidate = candidate.dropna(subset=["question", "answer"]).copy()
    candidate = candidate[candidate["question"].astype(str).str.strip().astype(bool)]
    candidate["source_name"] = name

    if verbose:
        print(f"\n{'='*72}\nEVALUATING: {name}  ({len(candidate):,} QA pairs)\n{'='*72}")

    # ---- 1/2. candidate alone: does it match, and what SHAPE is it? ----
    m_cand = GroundTruthMatcher(pool=candidate[["question", "answer", "source_name"]].reset_index(drop=True))
    feat_cand, _ = compute_gt_features(train, m_cand)
    rep_cand = _coverage_report(feat_cand, train, name)

    if verbose:
        print(f"\n--- {name} ALONE ---")
        print(f"  total matched: {rep_cand['total_matched']}/{len(train)}")
        for k in ["context", "no_context"]:
            r = rep_cand[k]
            print(f"  {k:11s} {r['matched']:3d}/{r['rows']:3d} = {r['rate']*100:5.1f}%   corr={r['corr']:.3f}"
                  if not np.isnan(r["corr"]) else
                  f"  {k:11s} {r['matched']:3d}/{r['rows']:3d} = {r['rate']*100:5.1f}%   corr=n/a")

    # ---- 3. incremental: the number that actually matters ----
    base = baseline_pool if baseline_pool is not None else load_all_sources()
    m_base = GroundTruthMatcher(pool=base)
    feat_base, _ = compute_gt_features(train, m_base)
    rep_base = _coverage_report(feat_base, train, "baseline")

    merged = pd.concat([base, candidate[["question", "answer", "source_name"]]], ignore_index=True)
    m_merged = GroundTruthMatcher(pool=merged)
    feat_merged, _ = compute_gt_features(train, m_merged)
    rep_merged = _coverage_report(feat_merged, train, "merged")

    delta_total = rep_merged["total_matched"] - rep_base["total_matched"]
    delta_noctx = rep_merged["no_context"]["matched"] - rep_base["no_context"]["matched"]
    delta_ctx = rep_merged["context"]["matched"] - rep_base["context"]["matched"]

    if verbose:
        print(f"\n--- INCREMENTAL (added to the existing {len(base):,}-pair pool) ---")
        print(f"  total matched:  {rep_base['total_matched']:3d} -> {rep_merged['total_matched']:3d}   ({delta_total:+d} rows)")
        print(f"  context:        {rep_base['context']['matched']:3d} -> {rep_merged['context']['matched']:3d}   ({delta_ctx:+d})")
        print(f"  no-context:     {rep_base['no_context']['matched']:3d} -> {rep_merged['no_context']['matched']:3d}   ({delta_noctx:+d})   <- the segment that matters")
        base_corr = rep_base["no_context"]["corr"]
        merged_corr = rep_merged["no_context"]["corr"]
        print(f"  no-ctx corr:    {base_corr:.3f} -> {merged_corr:.3f}   (must NOT drop -- a drop means false matches)")

        print(f"\n--- VERDICT ---")
        if delta_total == 0:
            print("  ✗ ADDS NOTHING. Same shape as something already in the pool. Skip it.")
        elif delta_noctx >= 10 and (np.isnan(merged_corr) or merged_corr >= base_corr - 0.03):
            print(f"  ✓ ADD IT. +{delta_noctx} no-context rows with no correlation loss -- this is the")
            print("    same signature bangla-mmlu had before it moved the real score +0.119.")
        elif delta_noctx >= 10:
            print(f"  ~ MAYBE. +{delta_noctx} no-context rows BUT correlation dropped "
                  f"({base_corr:.3f} -> {merged_corr:.3f}) -- likely injecting false matches.")
            print("    Try raising MATCH_THRESHOLD above 0.75 for this source before trusting it.")
        else:
            print(f"  ~ MARGINAL. Only {delta_total:+d} rows ({delta_noctx:+d} no-context). "
                  "Probably not worth the Phase-2 bundling cost.")

    return {"alone": rep_cand, "baseline": rep_base, "merged": rep_merged,
            "delta_total": delta_total, "delta_no_context": delta_noctx, "delta_context": delta_ctx}


if __name__ == "__main__":
    # Worked example: hishab/titulm-bangla-mmlu's Bengali_Grammar + Bengali_Literature configs.
    # Rationale for this specific candidate: 28 of the still-unmatched train rows are
    # `vocabulary` task-type and 3 are `grammar` -- these two configs are shaped like exactly
    # those rows, and nothing in the current pool is.
    import urllib.request
    import os

    CONFIGS = ["Bengali_Grammar", "Bengali_Literature"]
    parts = []
    for cfg in CONFIGS:
        for split in ["validation", "test"]:
            tmp = f"_titulm_{cfg}_{split}.parquet"
            url = (f"https://huggingface.co/datasets/hishab/titulm-bangla-mmlu/"
                   f"resolve/main/{cfg}/{split}-00000-of-00001.parquet")
            try:
                if not os.path.exists(tmp):
                    urllib.request.urlretrieve(url, tmp)
                parts.append(pd.read_parquet(tmp))
            except Exception as e:
                print(f"  skip {cfg}/{split}: {e}")

    if parts:
        raw = pd.concat(parts, ignore_index=True)
        print("titulm raw:", raw.shape, "| cols:", raw.columns.tolist())
        L2I = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        opt_col = "options" if "options" in raw.columns else "choices"

        def _resolve(row):
            opts = list(row[opt_col]) if row[opt_col] is not None else []
            a = str(row["answer"]).strip()
            i = L2I.get(a.upper())
            if i is not None and i < len(opts):
                return opts[i]
            return a if a else None  # some configs store the answer text directly

        raw["ans_text"] = raw.apply(_resolve, axis=1)
        cand = raw[["question", "ans_text"]].dropna().rename(columns={"ans_text": "answer"})
        evaluate_candidate(cand, name="titulm-grammar+literature")
