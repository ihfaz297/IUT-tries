import pandas as pd

sub = pd.read_csv("submission.csv")
print(f"Total: {len(sub)}")
print(f"Faithful (1): {sub['label'].sum()} ({sub['label'].sum()/len(sub)*100:.1f}%)")
print(f"Hallucinated (0): {(1-sub['label']).sum()} ({(1-sub['label']).sum()/len(sub)*100:.1f}%)")

# Compare with previous run
old = pd.read_csv("run_log.csv")
print("\n=== PREVIOUS RUN (run_log.csv) ===")
print(old.tail(1).to_string())

new = pd.read_csv("run_log (1).csv")
print("\n=== NEW RUN (run_log (1).csv) ===")
print(new.to_string())