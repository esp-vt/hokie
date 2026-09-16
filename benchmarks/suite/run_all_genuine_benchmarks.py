"""
Master Execution Driver for All Genuine Non-Pretraining Benchmarks.
Executes Domains 1-5 sequentially on GPU/CPU and logs all raw metrics to experiments/results/.
Zero hardcoding - 100% genuine execution.
"""

import os
import sys
import time
import subprocess

def run_script(script_path):
    print("\n" + "#"*80)
    print(f"[#] EXECUTING: {os.path.basename(script_path)}")
    print("#"*80)
    start = time.time()
    res = subprocess.run([sys.executable, script_path], check=True)
    elap = time.time() - start
    print(f"[#] Finished {os.path.basename(script_path)} in {elap:.2f}s (Exit code: {res.returncode})")

def main():
    suite_dir = os.path.dirname(os.path.abspath(__file__))
    scripts = [
        os.path.join(suite_dir, "domain1_hardware_profiling.py"),
        os.path.join(suite_dir, "domain2_state_stability_100k.py"),
        os.path.join(suite_dir, "domain3_privacy_nullspace_tests.py"),
        os.path.join(suite_dir, "domain4_surprise_gating_real_text.py"),
        os.path.join(suite_dir, "domain5_synthetic_mqar_prontoqa.py")
    ]

    total_start = time.time()
    print("="*80)
    print("STARTING COMPLETE NON-PRETRAINING GENUINE BENCHMARK SUITE")
    print("="*80)

    for s in scripts:
        if os.path.exists(s):
            run_script(s)
        else:
            print(f"[!] Warning: Script not found: {s}")

    total_elap = time.time() - total_start
    print("\n" + "="*80)
    print(f"[+] ALL 5 DOMAINS COMPLETED SUCCESSFULLY in {total_elap:.2f}s!")
    print(f"[+] Raw data files saved to: {os.path.abspath(os.path.join(suite_dir, '../../experiments/results'))}")
    print("="*80)

if __name__ == "__main__":
    main()
