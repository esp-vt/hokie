#!/usr/bin/env bash
set -e

echo "======================================================================"
echo "  Reproducing All NeuroWorld-LM Experiments & Figures (ICLR Suite)  "
echo "======================================================================"

PYTHON_BIN="/home/eun/miniforge3/envs/mura/bin/python"

echo "[0/7] Running Unit Test Suite (tests/run_all_tests.py)..."
$PYTHON_BIN tests/run_all_tests.py

echo "[1/7] Running Core Academic Benchmarks (MQAR 16k & PrOntoQA)..."
$PYTHON_BIN run_paper_experiments.py

echo "[2/7] Running Critical Reviewer Defense Suite (Drift, Noise, Pareto)..."
$PYTHON_BIN run_critical_defense.py

echo "[3/7] Running 100-Turn Persistent Dialogue Benchmark..."
$PYTHON_BIN benchmarks/multiturn_dialogue_eval.py

echo "[4/7] Running Comprehensive Ablation Grid (4 Axes, 12 Models)..."
$PYTHON_BIN run_ablation_grid.py

echo "[5/7] Generating Publication-Ready High-Res Figures (Figs 1..5)..."
$PYTHON_BIN visualize_mechanisms.py
$PYTHON_BIN benchmarks/hardware_profiling.py
$PYTHON_BIN visualize_needle_haystack.py

echo "[6/8] Running Real Natural Language Training (TinyStories & GSM8K)..."
$PYTHON_BIN train_tinystories.py
$PYTHON_BIN train_gsm8k_reasoning.py

echo "[7/9] Running Real NLP Academic Benchmarks (ARC-Challenge, OpenBookQA, DailyDialog)..."
$PYTHON_BIN benchmarks/real_nlp_benchmarks.py

echo "[8/10] Running Drawback Mitigation Experiments (Anchored Rollout, Thought Probing, Curriculum)..."
$PYTHON_BIN benchmarks/mitigation_experiments.py

echo "[9/10] Running Advanced Rigorous Suite (LLM-Judge, Difficulty vs Depth, Causal Intervention)..."
$PYTHON_BIN benchmarks/advanced_rigorous_benchmarks.py

echo "[10/10] Running Active Semantic Forgetting Engine (ASFE) Suite..."
$PYTHON_BIN benchmarks/active_forgetting_eval.py

echo "======================================================================"
echo "  [SUCCESS] All Experiments, Figures, and Datasets Reproduced!  "
echo "======================================================================"
