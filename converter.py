"""
Converter: Merges joggota_core.py + submission_pipeline.py into a single
self-contained Kaggle notebook. No external .py imports needed.

Usage:
  python converter.py
"""
import json

# Read both source files (normalize line endings for Windows compat)
with open("joggota_core.py", "r", encoding="utf-8") as f:
    joggota_code = f.read().replace("\r\n", "\n")

with open("submission_pipeline.py", "r", encoding="utf-8") as f:
    pipeline_code = f.read().replace("\r\n", "\n")

# Remove the import line from pipeline since we're inlining joggota_core
pipeline_code = pipeline_code.replace(
    "from joggota_core import extract_joggota_features\n", ""
)
pipeline_code = pipeline_code.replace(
    "from joggota_core import extract_joggota_features", ""
)

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# অলীকবচন — Joggota + BanglaBERT Pipeline\n",
                "**Self-contained notebook.** No external `.py` files needed.\n\n",
                "### Setup Checklist:\n",
                "1. GPU: **T4 x2** or **P100**\n",
                "2. Attach fine-tuned BanglaBERT-large weights as a Kaggle Dataset → path: `/kaggle/input/banglabert-finetuned-hallu`\n",
                "3. Attach mDeBERTa weights as a Kaggle Dataset → update `NLI_MODEL_NAME` below\n", 
                "4. Attach LaBSE weights as a Kaggle Dataset → update `EMBED_MODEL_NAME` below\n",
                "5. Upload `offline_corpus.json` alongside this notebook (or as a Dataset)\n",
                "6. Upload `dataset samples.json` and `test set.csv`\n",
                "7. For Phase 2: toggle **Internet OFF** before running"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q accelerate sentence-transformers xgboost\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Part 1: Joggota Engine (Deterministic Rules + Mini-RAG)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in joggota_code.split("\n")]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Part 2: Train BanglaBERT (Auto-saves to /kaggle/working/banglabert_finetuned/)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Auto-training cell: fine-tunes BanglaBERT-large on 299 labeled samples + synthetic pairs\n",
                "import os, gc, json, random, re\n",
                "import numpy as np, pandas as pd\n",
                "import torch, torch.nn as nn, torch.nn.functional as F\n",
                "from torch.utils.data import Dataset, DataLoader\n",
                "from sklearn.metrics import f1_score\n",
                "from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup\n",
                "\n",
                "SEED = 42\n",
                "random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)\n",
                "if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)\n",
                "DEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
                "MODEL_NAME = \"csebuetnlp/banglabert_large\"\n",
                "BATCH_SIZE, EPOCHS, LR, MAX_LEN, WARMUP = 8, 3, 1e-5, 256, 0.1\n",
                "CKPT_DIR = \"/kaggle/working/banglabert_finetuned\"\n",
                "\n",
                "# ---- Load & augment ----\n",
                "with open(\"dataset samples.json\", \"r\", encoding=\"utf-8\") as f: data = json.load(f)\n",
                "train_df = pd.DataFrame(data)\n",
                "premises, responses, labels = [], [], []\n",
                "for _, r in train_df.iterrows():\n",
                "    ctx = r.get(\"context\", \"\")\n",
                "    if pd.isna(ctx) or str(ctx).strip().lower() in (\"[null]\", \"null\", \"none\", \"nan\", \"\"):\n",
                "        premise = str(r[\"prompt_bn\"])\n",
                "    else: premise = str(r[\"prompt_bn\"]) + \" \" + str(ctx).strip()\n",
                "    premises.append(premise); responses.append(str(r[\"response_bn\"])); labels.append(int(r[\"label\"]))\n",
                "\n",
                "pos_idx = [i for i, l in enumerate(labels) if l == 1]\n",
                "neg_idx = [i for i, l in enumerate(labels) if l == 0]\n",
                "aug_premises, aug_responses, aug_labels = list(premises), list(responses), list(labels)\n",
                "random.shuffle(pos_idx)\n",
                "for i, pi in enumerate(pos_idx):\n",
                "    nj = neg_idx[i % len(neg_idx)]\n",
                "    aug_premises.append(premises[pi]); aug_responses.append(responses[nj]); aug_labels.append(0)\n",
                "random.shuffle(neg_idx)\n",
                "for i, ni in enumerate(neg_idx):\n",
                "    pj = pos_idx[i % len(pos_idx)]\n",
                "    aug_premises.append(premises[ni]); aug_responses.append(responses[pj]); aug_labels.append(1)\n",
                "print(f\"Augmented: {len(aug_labels)} pairs ({sum(aug_labels)} faithful, {len(aug_labels)-sum(aug_labels)} hallucinated)\")\n",
                "\n",
                "# ---- Dataset ----\n",
                "class TrainPairDS(Dataset):\n",
                "    def __init__(self, prems, resps, ys, tok, mx=MAX_LEN):\n",
                "        self.p=prems; self.h=resps; self.y=ys; self.t=tok; self.m=mx\n",
                "    def __len__(self): return len(self.p)\n",
                "    def __getitem__(self, i):\n",
                "        e = self.t(self.p[i], self.h[i], truncation=True, max_length=self.m, padding=\"max_length\", return_tensors=\"pt\")\n",
                "        return {\"input_ids\": e[\"input_ids\"].squeeze(0), \"attention_mask\": e[\"attention_mask\"].squeeze(0), \"labels\": torch.tensor(self.y[i], dtype=torch.long)}\n",
                "\n",
                "class FocalLoss(nn.Module):\n",
                "    def __init__(self, gamma=2.0, alpha=1.0):\n",
                "        super().__init__(); self.g = gamma; self.register_buffer(\"w\", torch.tensor([alpha, 1.0]))\n",
                "    def forward(self, lg, y):\n",
                "        ce = F.cross_entropy(lg, y, weight=self.w.to(lg.device), reduction=\"none\"); pt = torch.exp(-ce); return ((1-pt)**self.g*ce).mean()\n",
                "\n",
                "# ---- Train ----\n",
                "from sklearn.model_selection import train_test_split\n",
                "tp, vp, tr, vr, ty, vy = train_test_split(aug_premises, aug_responses, aug_labels, test_size=0.2, random_state=42, stratify=aug_labels)\n",
                "tok = AutoTokenizer.from_pretrained(MODEL_NAME)\n",
                "model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True).float().to(DEVICE)\n",
                "tld = DataLoader(TrainPairDS(tp, tr, ty, tok), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)\n",
                "vld = DataLoader(TrainPairDS(vp, vr, vy, tok), batch_size=BATCH_SIZE*2, shuffle=False, pin_memory=True)\n",
                "crit = FocalLoss(2.0, 1.0); opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)\n",
                "tot = len(tld)*EPOCHS; sch = get_cosine_schedule_with_warmup(opt, int(tot*WARMUP), tot)\n",
                "scaler = torch.amp.GradScaler(\"cuda\"); best_f1, best_state = 0.0, None\n",
                "\n",
                "@torch.no_grad()\n",
                "def evalfn(m, ld):\n",
                "    m.eval(); ap, al = [], []\n",
                "    for b in ld:\n",
                "        with torch.amp.autocast(\"cuda\", dtype=torch.float16): lg = m(input_ids=b[\"input_ids\"].to(DEVICE), attention_mask=b[\"attention_mask\"].to(DEVICE)).logits.float()\n",
                "        pr = torch.softmax(lg, -1)[:,1].cpu().numpy(); ap.extend((pr>=0.5).astype(int)); al.extend(b[\"labels\"].numpy())\n",
                "    return f1_score(al, ap)\n",
                "\n",
                "print(f\"Training {EPOCHS} epochs...\")\n",
                "for ep in range(EPOCHS):\n",
                "    model.train(); tl=0.0\n",
                "    for b in tld:\n",
                "        opt.zero_grad()\n",
                "        with torch.amp.autocast(\"cuda\", dtype=torch.float16): lg = model(input_ids=b[\"input_ids\"].to(DEVICE), attention_mask=b[\"attention_mask\"].to(DEVICE)).logits; lo = crit(lg, b[\"labels\"].to(DEVICE))\n",
                "        scaler.scale(lo).backward(); scaler.step(opt); scaler.update(); sch.step(); tl += lo.item()\n",
                "    vf = evalfn(model, vld)\n",
                "    print(f\"  Epoch {ep+1}: loss={tl/len(tld):.4f} | val F1={vf:.4f}\")\n",
                "    if vf > best_f1: best_f1 = vf; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}\n",
                "\n",
                "if best_state is not None: model.load_state_dict(best_state)\n",
                "os.makedirs(CKPT_DIR, exist_ok=True)\n",
                "model.save_pretrained(CKPT_DIR); tok.save_pretrained(CKPT_DIR)\n",
                "print(f\"\\n✅ Saved fine-tuned model to {CKPT_DIR}/ (val F1={best_f1:.4f})\")\n",
                "del model, tok, tld, vld; gc.collect(); torch.cuda.empty_cache()\n",
                "print(\"Training complete. Ready for inference.\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Part 3: Submission Pipeline (NLI + BanglaBERT + XGBoost)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in pipeline_code.split("\n")]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Part 4: Run Inference"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "train_and_predict()\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("submission_kaggle.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Created submission_kaggle.ipynb (self-contained, no external imports)")