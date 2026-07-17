import json, random
random.seed(42)

data = json.load(open("benhallueval_training.json", "r", encoding="utf-8"))
faithful = [d for d in data if d["label"] == 1]
hallucinated = [d for d in data if d["label"] == 0]

print("=" * 60)
print("FAITHFUL (label=1) — 10 random samples")
print("=" * 60)
for s in random.sample(faithful, 10):
    print(f"Q: {s['prompt_bn'][:80]}")
    print(f"A: {s['response_bn'][:80]}")
    print()

print("=" * 60)
print("HALLUCINATED (label=0) — 10 random samples")
print("=" * 60)
for s in random.sample(hallucinated, 10):
    print(f"Q: {s['prompt_bn'][:80]}")
    print(f"A: {s['response_bn'][:80]}")
    print()