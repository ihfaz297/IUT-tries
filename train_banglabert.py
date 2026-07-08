"""
Lightweight BanglaBERT fine-tuning script.
Uses the 299 labeled samples + synthetic negative pairs.
Saves checkpoint to banglabert_checkpoint.pt for the pipeline to auto-detect.
"""
import os, gc, json, random, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup
)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "csebuetnlp/banglabert_large"
CHECKPOINT_OUT = "banglabert_checkpoint.pt"
BATCH_SIZE = 8
EPOCHS = 3
LR = 1e-5
MAX_LEN = 256
WARMUP = 0.1

# ---------------------------------------------------------------------------
# 1. Load competition data
# ---------------------------------------------------------------------------
print("Loading data...")
with open("dataset samples.json", "r", encoding="utf-8") as f:
    data = json.load(f)
train_df = pd.DataFrame(data)

# Split prompt/response into premise/hypothesis pairs
premises = []
responses = []
labels = []
for _, r in train_df.iterrows():
    ctx = r.get("context", "")
    if pd.isna(ctx) or str(ctx).strip().lower() in ("[null]", "null", "none", "nan", ""):
        premise = str(r["prompt_bn"])
    else:
        premise = str(r["prompt_bn"]) + " " + str(ctx).strip()
    premises.append(premise)
    responses.append(str(r["response_bn"]))
    labels.append(int(r["label"]))

# Generate synthetic negatives: swap faithful responses with hallucinated ones
# This doubles the training set and adds more contrast
pos_idx = [i for i, l in enumerate(labels) if l == 1]
neg_idx = [i for i, l in enumerate(labels) if l == 0]

aug_premises = list(premises)
aug_responses = list(responses)
aug_labels = list(labels)

# For each positive, pair with a random hallucinated response (extrinsic hallucination)
random.shuffle(pos_idx)
for i, pi in enumerate(pos_idx):
    nj = neg_idx[i % len(neg_idx)]
    aug_premises.append(premises[pi])
    aug_responses.append(responses[nj])
    aug_labels.append(0)

# For each negative, pair with a random faithful response (adds challenging negatives)
random.shuffle(neg_idx)
for i, ni in enumerate(neg_idx):
    pj = pos_idx[i % len(pos_idx)]
    aug_premises.append(premises[ni])
    aug_responses.append(responses[pj])
    aug_labels.append(1)

print(f"Original: {len(labels)} pairs ({sum(labels)} faithful, {len(labels)-sum(labels)} hallucinated)")
print(f"Augmented: {len(aug_labels)} pairs ({sum(aug_labels)} faithful, {len(aug_labels)-sum(aug_labels)} hallucinated)")

# ---------------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------------
class PairDataset(Dataset):
    def __init__(self, premises, responses, labels, tokenizer, max_len=MAX_LEN):
        self.premises = premises
        self.responses = responses
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.premises)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.premises[i], self.responses[i],
            truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[i], dtype=torch.long),
        }

# ---------------------------------------------------------------------------
# 3. Focal Loss (same as banglabert-train-m.ipynb)
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", torch.tensor([alpha, 1.0]))

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight.to(logits.device), reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------
print(f"\nLoading {MODEL_NAME} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
).float().to(DEVICE)

# Split into train/val (80/20 — stratified)
from sklearn.model_selection import train_test_split
train_prems, val_prems, train_resps, val_resps, train_y, val_y = train_test_split(
    aug_premises, aug_responses, aug_labels,
    test_size=0.2, random_state=SEED, stratify=aug_labels
)

train_ds = PairDataset(train_prems, train_resps, train_y, tokenizer)
val_ds = PairDataset(val_prems, val_resps, val_y, tokenizer)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, pin_memory=True)

criterion = FocalLoss(gamma=2.0, alpha=1.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * WARMUP), total_steps)
scaler = torch.amp.GradScaler("cuda")

best_f1 = 0.0
best_state = None

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE)
            ).logits.float()
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_preds.extend((probs >= 0.5).astype(int))
        all_labels.extend(batch["labels"].numpy())
    return f1_score(all_labels, all_preds)

print(f"\nTraining {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE)
            ).logits
            loss = criterion(logits, batch["labels"].to(DEVICE))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()

    val_f1 = evaluate(model, val_loader)
    print(f"  Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f} | val F1={val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

# ---------------------------------------------------------------------------
# 5. Save checkpoint
# ---------------------------------------------------------------------------
if best_state is not None:
    model.load_state_dict(best_state)

checkpoint = {
    "model_state_dict": model.state_dict(),
    "config": {
        "model_name": MODEL_NAME,
        "val_f1": best_f1,
        "num_labels": 2,
    }
}
torch.save(checkpoint, CHECKPOINT_OUT)
print(f"\n✅ Saved checkpoint to {CHECKPOINT_OUT} (val F1={best_f1:.4f})")

# Also save the full model for HuggingFace compatibility
model_dir = "banglabert_finetuned"
os.makedirs(model_dir, exist_ok=True)
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
print(f"✅ Saved full model to {model_dir}/")

# Cleanup
del model, tokenizer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("Done.")