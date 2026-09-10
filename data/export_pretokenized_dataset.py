#!/usr/bin/env python3
import os
import torch
from data.hf_dataset_loader import RealCorpusDataLoader

def export():
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    loader = RealCorpusDataLoader("roneneldan/TinyStories")
    os.makedirs("data/cache", exist_ok=True)

    train_path = "data/cache/tinystories_train_tokens.pt"
    val_path = "data/cache/tinystories_val_tokens.pt"

    torch.save(loader.train_tokens, train_path)
    torch.save(loader.val_tokens, val_path)

    print(f"[✓] Successfully exported pre-tokenized data:")
    print(f"    Train: {train_path} ({len(loader.train_tokens):,} tokens, {os.path.getsize(train_path)/1024/1024:.2f} MB)")
    print(f"    Val  : {val_path} ({len(loader.val_tokens):,} tokens, {os.path.getsize(val_path)/1024/1024:.2f} MB)")

if __name__ == "__main__":
    export()
