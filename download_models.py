import os
from huggingface_hub import snapshot_download

def download_model(repo_id, local_dir):
    print(f"Downloading {repo_id} to {local_dir}...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "coreml/*"],
        local_dir_use_symlinks=False
    )
    print(f"Successfully downloaded {repo_id} to {local_dir}")

if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "offline_models")
    os.makedirs(base_dir, exist_ok=True)

    models = [
        ("Qwen/Qwen2.5-1.5B-Instruct", "qwen-2.5-1.5b-instruct"),
        ("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", "mdeberta-v3-base-mnli-xnli"),
        ("sentence-transformers/LaBSE", "labse")
    ]

    for hf_repo, folder_name in models:
        local_path = os.path.join(base_dir, folder_name)
        if not os.path.exists(local_path):
            download_model(hf_repo, local_path)
        else:
            print(f"Model {folder_name} already exists at {local_path}. Skipping.")
    print("All models downloaded for offline Kaggle usage.")
