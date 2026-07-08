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
                "# অলীকবচন — Joggota + TigerLLM Pipeline\n",
                "**Self-contained notebook.** No external `.py` files needed.\n\n",
                "### Setup Checklist:\n",
                "1. GPU: **T4 x2**\n",
                "2. Attach TigerLLM-9B weights as a Kaggle Dataset → update `TIGERLLM_MODEL_NAME` below\n",
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
                "!pip install -q bitsandbytes accelerate sentence-transformers xgboost\n"
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
                "## Part 2: Submission Pipeline (NLI + TigerLLM + XGBoost)"
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
                "## Part 3: Run Inference"
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