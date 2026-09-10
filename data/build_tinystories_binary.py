#!/usr/bin/env python3
"""
Direct Arrow reader to binary PyTorch token tensors.
Reads local arrow cache without HuggingFace Hub network dependency.
"""
import os
import pyarrow.ipc as ipc
import torch
from transformers import AutoTokenizer

def build():
    print("[*] Loading local gpt2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    arrow_path = "/home/eun/.cache/huggingface/datasets/roneneldan___tiny_stories/default/0.0.0/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/tiny_stories-validation.arrow"
    print(f"[*] Reading arrow file: {arrow_path}")
    table = ipc.open_stream(arrow_path).read_all()
    texts = table["text"].to_pylist()
    print(f"[✓] Total available stories in arrow file: {len(texts):,}")

    # Split into 3,500 train stories and 500 validation stories
    train_stories = texts[:3500]
    val_stories = texts[3500:4000]

    print(f"[*] Tokenizing {len(train_stories)} train stories...")
    train_tokens = []
    for s in train_stories:
        if s and len(s.strip()) > 0:
            train_tokens.extend(tokenizer.encode(s, add_special_tokens=True))
    train_tensor = torch.tensor(train_tokens, dtype=torch.long)

    print(f"[*] Tokenizing {len(val_stories)} val stories...")
    val_tokens = []
    for s in val_stories:
        if s and len(s.strip()) > 0:
            val_tokens.extend(tokenizer.encode(s, add_special_tokens=True))
    val_tensor = torch.tensor(val_tokens, dtype=torch.long)

    os.makedirs("data/cache", exist_ok=True)
    train_out = "data/cache/tinystories_train_tokens.pt"
    val_out = "data/cache/tinystories_val_tokens.pt"

    torch.save(train_tensor, train_out)
    torch.save(val_tensor, val_out)

    print(f"[✓] Saved Train Tensor: {train_out} ({len(train_tensor):,} tokens, {os.path.getsize(train_out)/1024/1024:.2f} MB)")
    print(f"[✓] Saved Val Tensor  : {val_out} ({len(val_tensor):,} tokens, {os.path.getsize(val_out)/1024/1024:.2f} MB)")

if __name__ == "__main__":
    build()
