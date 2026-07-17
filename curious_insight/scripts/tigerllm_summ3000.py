import subprocess, sys
subprocess.run([sys.executable, "scripts/evaluate_tigerllm.py", "--task", "summ3000"])
