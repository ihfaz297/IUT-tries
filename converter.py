import json

with open("submission_pipeline.py", "r", encoding="utf-8") as f:
    code = f.read()

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Joggota + TigerLLM Kaggle Pipeline\n",
                "This notebook is ready to be run offline on Kaggle. Make sure to attach the TigerLLM dataset!"
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
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code.split("\n")]
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
