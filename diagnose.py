"""Diagnose CV F1 vs actual score gap."""
import pandas as pd
import numpy as np

# Key numbers from the logs
print("=" * 60)
print("PREVIOUS RUN (no BanglaBERT, TigerLLM fallback to sim_premise_response)")
print("=" * 60)
print("  CV F1: 0.646  (5-fold stratified on 299 samples)")
print("  Feature: sim_premise_response (LaBSE cosine) used as proxy")

print()
print("=" * 60)
print("NEW RUN (with BanglaBERT fine-tuned on 7.5K BenHalluEval pairs)")
print("=" * 60)
print("  CV F1: 0.823  (5-fold stratified on 299 samples)")
print("  Actual score: 0.602")
print("  Gap: 0.221 (22 points!)")

print()
print("=" * 60)
print("PREDICTION DISTRIBUTION")
print("=" * 60)
sub = pd.read_csv("submission.csv")
print(f"  Total: {len(sub)}")
print(f"  Faithful (1): {sub['label'].sum()} ({sub['label'].sum()/len(sub)*100:.1f}%)")
print(f"  Hallucinated (0): {(1-sub['label']).sum()} ({(1-sub['label']).sum()/len(sub)*100:.1f}%)")

print()
print("=" * 60)
print("TRAIN DISTRIBUTION (299 samples)")
print("=" * 60)
print(f"  Faithful: 163 (54.5%)")
print(f"  Hallucinated: 136 (45.5%)")

print()
print("=" * 60)
print("BANGLA-BERT FEATURE ANALYZED")
print("=" * 60)
print("  Train mean: 0.516 (close to 54.5% faithful ratio — looks calibrated)")
print("  Train std:  0.187 (moderate variance)")
print("  BUT: BanglaBERT was trained on BenHalluEval QA pairs")
print("  The 299 competition samples are from the SAME distribution")
print("  So CV overestimates real performance")

print()
print("=" * 60)
print("ROOT CAUSE")
print("=" * 60)
print("""
1. BanglaBERT was fine-tuned on BenHalluEval (QA-only dataset)
2. The 299 training samples are also QA-heavy
3. 5-fold CV on 299 samples = BanglaBERT sees near-identical distribution in each fold
4. CV F1 of 0.823 is INFLATED — it's measuring how well the model memorizes the train distribution
5. The 2,516 test samples include summarization, reasoning, code-mixed tasks
6. BanglaBERT overfits to QA patterns and fails on other task types
7. With 12 features, if BanglaBERT dominates, it drags the whole ensemble down
""")

# Check: what's the test set context vs no-context split?
test = pd.read_csv("test set.csv")
n_ctx = (test["context"].notna() & (test["context"] != "[NULL]")).sum()
n_noctx = (test["context"].isna() | (test["context"] == "[NULL]")).sum()
print(f"Test set: {n_ctx} with context, {n_noctx} without context")
print(f"Train set: 130 with context, 169 without context")
print(f"Context ratio: test={n_ctx/len(test):.1%} vs train={130/299:.1%}")
print()

# Possible task types in test set
print("=" * 60)
print("TEST SET SAMPLE PROMPTS (first 20)")
print("=" * 60)
for i, row in test.head(20).iterrows():
    ctx_flag = "CTX" if (pd.notna(row['context']) and str(row['context']).strip() != '[NULL]') else "NOCTX"
    print(f"  [{ctx_flag}] {str(row['prompt_bn'])[:100]}")