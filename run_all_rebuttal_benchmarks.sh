#!/usr/bin/env bash
# ==============================================================================
# NeuroWorld-LM: Master Rebuttal & Defense Benchmark Execution Suite
# Run this script once GPU becomes available to generate all empirical figures,
# validation logs, and ISO-FLOP comparative evaluation reports.
# ==============================================================================
set -e

DEVICE="cuda"
if ! command -v nvidia-smi &> /dev/null; then
    echo "[!] No NVIDIA GPU detected. Falling back to CPU..."
    DEVICE="cpu"
fi

echo "========================================================================"
echo "  Executing NeuroWorld-LM Rebuttal Defense Benchmark Suite on: $DEVICE  "
echo "========================================================================"

PYTHON_BIN=$(which python3 || which python)

# 1. Adversarial Defense Suite (Refuting Non-linear Probe, Typos, 100k Key, ECE)
echo -e "\n[1/5] Running Adversarial Defense Suite (Refutations 1, 2, 4, 5)..."
$PYTHON_BIN benchmarks/adversarial_defense_suite.py

# 2. Transformer Killer Suite (CD-NIAH, Ephemeral Scratchpad, 100k Multi-Topic)
echo -e "\n[2/5] Running Transformer-Killer Benchmark Suite..."
$PYTHON_BIN benchmarks/transformer_killer_eval.py

# 3. Rigorous Multi-Rank PII Unlearning & Adversarial Threat Suite
echo -e "\n[3/5] Running Rigorous Multi-Rank PII Unlearning Suite..."
$PYTHON_BIN benchmarks/rigorous_pii_unlearning_eval.py

# 4. ISO-FLOP Comparative Pre-Training & Memory Benchmark (vs LLaMA-style Transformer++)
echo -e "\n[4/5] Running Strict ISO-FLOP Comparative Benchmark..."
$PYTHON_BIN benchmarks/run_iso_flop_benchmark.py --device $DEVICE --train_steps 100 --seq_len 512 --batch_size 4

# 5. KV Cache Recomputation vs Standard KV Cache vs NeuroWorld-LM Benchmark
echo -e "\n[5/5] Running KV Cache Recomputation vs NeuroWorld-LM Benchmark..."
$PYTHON_BIN benchmarks/recompute_vs_kvcache_eval.py --device $DEVICE --prompt_len 1024 --gen_tokens 128 --batch_size 2

echo -e "\n========================================================================"
echo "  [✓] All Rebuttal Benchmarks Successfully Executed & Reports Updated!  "
echo "========================================================================"
