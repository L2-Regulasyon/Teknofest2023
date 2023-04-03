# HF Bert Model Upload Script
from huggingface_hub import HfApi

local_path = f'../checkpoint/toxic-dbmdz-bert-base-turkish-128k-uncased/'

api = HfApi()
repo = "toxic-dbmdz-bert-base-turkish-128k-uncased"

files_to_push_to_hub = [
    'config.json',
    'pytorch_model.bin',
    'special_tokens_map.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'vocab.txt'
]

for filename in files_to_push_to_hub:
    api.upload_file(
        path_or_fileobj=f"{local_path}{filename}",
        repo_id=f"l2reg/{repo}",
        path_in_repo=filename,
        repo_type="model",
        create_pr=1
    )
