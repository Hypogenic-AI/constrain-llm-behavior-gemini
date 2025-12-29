from datasets import load_dataset
import os

DATASETS = [
    ("truthful_qa", "generation", "datasets/truthful_qa"),
    ("squad_v2", None, "datasets/squad_v2"),
    ("nq_open", None, "datasets/nq_open")
]

for name, config, path in DATASETS:
    print(f"Downloading {name}...")
    try:
        if config:
            ds = load_dataset(name, config)
        else:
            ds = load_dataset(name)
        ds.save_to_disk(path)
        print(f"Saved {name} to {path}")
    except Exception as e:
        print(f"Failed to download {name}: {e}")
