import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

class RealCorpusDataLoader:
    """
    HuggingFace Real Dataset Loader with BPE/Llama-compatible Tokenizer.
    Supports:
    - 'wikitext-2-raw-v1'
    - 'roneneldan/TinyStories'
    - 'openai/gsm8k'
    """
    def __init__(
        self,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        tokenizer_name: str = "gpt2",
        seq_len: int = 128,
        batch_size: int = 8
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.seq_len = seq_len
        self.batch_size = batch_size

        print(f"[*] Initializing Tokenizer: {tokenizer_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.vocab_size = len(self.tokenizer)
        print(f"[*] Tokenizer Loaded | Vocabulary Size: {self.vocab_size:,}")

        # Check for pre-tokenized binary cache first
        cache_dir = os.path.join(os.path.dirname(__file__), "cache")
        cached_train = os.path.join(cache_dir, "tinystories_train_tokens.pt")
        cached_val = os.path.join(cache_dir, "tinystories_val_tokens.pt")

        if "TinyStories" in dataset_name and os.path.exists(cached_train) and os.path.exists(cached_val):
            print(f"[*] Found pre-tokenized binary cache at: {cache_dir}")
            self.train_tokens = torch.load(cached_train)
            self.val_tokens = torch.load(cached_val)
            print(f"[✓] Instant Load Ready | Train Tokens: {len(self.train_tokens):,} | Val Tokens: {len(self.val_tokens):,}")
            return

        print(f"[*] Loading HuggingFace Dataset: {dataset_name} ({dataset_config})")
        if "wikitext" in dataset_name:
            repo_id = "Salesforce/wikitext"
            self.raw_dataset = load_dataset(repo_id, dataset_config, split="train[:5000]")
            self.val_dataset = load_dataset(repo_id, dataset_config, split="validation[:500]")
            text_key = "text"
        elif "TinyStories" in dataset_name:
            repo_id = "roneneldan/TinyStories"
            self.raw_dataset = load_dataset(repo_id, split="train[:3000]")
            self.val_dataset = load_dataset(repo_id, split="validation[:300]")
            text_key = "text"
        else:
            self.raw_dataset = load_dataset(dataset_name, split="train[:2000]")
            self.val_dataset = load_dataset(dataset_name, split="test[:200]")
            text_key = "question"

        print(f"[*] Tokenizing Train Split ({len(self.raw_dataset)} samples)...")
        self.train_tokens = self._tokenize_and_flatten(self.raw_dataset, text_key)
        self.val_tokens = self._tokenize_and_flatten(self.val_dataset, text_key)
        print(f"[✓] Data Prep Ready | Train Tokens: {len(self.train_tokens):,} | Val Tokens: {len(self.val_tokens):,}")

    def _tokenize_and_flatten(self, dataset, text_key: str):
        all_ids = []
        for item in dataset:
            text = item[text_key]
            if text and len(text.strip()) > 0:
                ids = self.tokenizer.encode(text, add_special_tokens=True)
                all_ids.extend(ids)
        return torch.tensor(all_ids, dtype=torch.long)

    def get_batch(self, split: str = "train"):
        data = self.train_tokens if split == "train" else self.val_tokens
        max_idx = len(data) - self.seq_len - 1
        if max_idx <= 0:
            raise ValueError("Dataset too small for requested sequence length.")

        starts = torch.randint(0, max_idx, (self.batch_size,))
        x = torch.stack([data[i : i + self.seq_len] for i in starts])
        y = torch.stack([data[i + 1 : i + self.seq_len + 1] for i in starts])
        return x, y
