#!/usr/bin/env python3
"""
Master Pipeline Runner for Drug-Target Interaction (DTI) Benchmark (run_all.py)

Executes the full pipeline end-to-end in order:
1. filter_bindingdb.py  -> bindingdb_human_ki_kd.tsv
2. curate_dti_binary.py -> bindingdb_human_dti_binary.tsv
3. features.py          -> features/drug_features.npy, protein_features.npy, dti_features.npz
4. splits.py            -> splits/*.npz (4 generalization splits)
5. pytest               -> tests/test_splits.py (split validity assertions)
6. train.py             -> results/four_regime_table.csv & models (passes --train-size)
7. analyse.py           -> results/memorisation.csv, auc_vs_similarity.png, etc.

Each step is automatically skipped if its expected output already exists,
unless the `--force` flag is provided. (Steps 1 and 2 are skipped even with
--force because their curated outputs are final).
"""

import os
import sys
import time
import argparse
import subprocess


def run_command(cmd: list[str], step_name: str) -> None:
    """Run a subprocess command and stream output in real time; raise on failure."""
    print(f"\n{'=' * 80}")
    print(f"RUNNING STEP: {step_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")
    t0 = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"\n[ERROR] Step '{step_name}' failed with return code {res.returncode} after {elapsed:.2f}s.")
        sys.exit(res.returncode)
    print(f"\n[DONE] Step '{step_name}' completed successfully in {elapsed:.2f}s.\n")


def check_outputs_exist(outputs: list[str]) -> bool:
    """Check if all expected output paths exist."""
    return all(os.path.exists(p) for p in outputs)


def main():
    parser = argparse.ArgumentParser(
        description="Master Pipeline Runner for DTI 4-Regime Benchmark"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-running of all steps regardless of existing outputs",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=150000,
        help="Subsample training sample size per split passed to train.py (default: 150000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed (default: 42)",
    )
    args = parser.parse_args()

    t_start = time.time()
    py = sys.executable

    print("\n" + "#" * 80)
    print("STARTING DTI BENCHMARK PIPELINE (run_all.py)")
    print(f"Python interpreter: {py}")
    print(f"Force re-run:       {args.force}")
    print(f"Train sample size:  {args.train_size:,}")
    print("#" * 80 + "\n")

    # Step 1: filter_bindingdb.py (Outputs are final, skipped even with --force if present)
    step1_outputs = ["bindingdb_human_ki_kd.tsv"]
    if os.path.exists("bindingdb_human_ki_kd.tsv") or os.path.exists("bindingdb_human_filtered.tsv"):
        print("[SKIP] Step 1: filter_bindingdb.py - Filtered human binding affinity file already exists (final).")
    else:
        if not os.path.exists("BindingDB_All.tsv"):
            print("[ERROR] Neither 'BindingDB_All.tsv' nor filtered human dataset was found.")
            sys.exit(1)
        run_command([py, "filter_bindingdb.py"], "1. Filter BindingDB (Human, Valid SMILES/Sequence/Affinity)")

    # Step 2: curate_dti_binary.py (Outputs are final, skipped even with --force if present)
    step2_outputs = ["bindingdb_human_dti_binary.tsv"]
    if check_outputs_exist(step2_outputs):
        print("[SKIP] Step 2: curate_dti_binary.py - 'bindingdb_human_dti_binary.tsv' already exists (final).")
    else:
        run_command([py, "curate_dti_binary.py"], "2. Curate Binary DTI Pairs (<=100nM vs >=10uM)")

    # Step 3: features.py
    step3_outputs = [
        os.path.join("features", "drug_features.npy"),
        os.path.join("features", "protein_features.npy"),
        os.path.join("features", "dti_features.npz"),
    ]
    if not args.force and check_outputs_exist(step3_outputs):
        print("[SKIP] Step 3: features.py - Precomputed features in 'features/' already exist.")
    else:
        run_command([py, "features.py"], "3. Compute ECFP4 Fingerprints & AAC/DPC Protein Features")

    # Step 4: splits.py
    step4_outputs = [
        os.path.join("splits", "random.npz"),
        os.path.join("splits", "unseen_drugs.npz"),
        os.path.join("splits", "unseen_proteins.npz"),
        os.path.join("splits", "unseen_both.npz"),
    ]
    if not args.force and check_outputs_exist(step4_outputs):
        print("[SKIP] Step 4: splits.py - Four benchmark splits in 'splits/' already exist.")
    else:
        run_command([py, "splits.py"], "4. Generate 4 Benchmark Splits (Random, Unseen Drugs/Proteins/Both)")

    # Step 5: pytest
    print("\n" + "-" * 80)
    print("STEP 5: Running Unit & Split Integrity Tests (pytest)")
    print("-" * 80)
    run_command([py, "-m", "pytest", "tests/"], "5. Pytest Split & Feature Validation")

    # Step 6: train.py
    step6_outputs = [
        os.path.join("results", "four_regime_table.csv"),
        os.path.join("results", "four_regime_chart.png"),
        os.path.join("results", "rf_split1.joblib"),
    ]
    if not args.force and check_outputs_exist(step6_outputs):
        print(f"[SKIP] Step 6: train.py - Training outputs in 'results/' already exist. (Use --force to retrain)")
    else:
        cmd_train = [py, "train.py", "--train-size", str(args.train_size), "--seed", str(args.seed)]
        run_command(cmd_train, f"6. Train & Evaluate Models Across 4 Regimes (train-size={args.train_size:,})")

    # Step 7: analyse.py
    step7_outputs = [
        os.path.join("results", "memorisation.csv"),
        os.path.join("results", "worst_predictions.csv"),
        os.path.join("results", "worst_false_positives.csv"),
        os.path.join("results", "auc_vs_similarity.png"),
    ]
    if not args.force and check_outputs_exist(step7_outputs):
        print("[SKIP] Step 7: analyse.py - Diagnostic analysis in 'results/' already exists.")
    else:
        run_command([py, "analyse.py", "--train-size", str(args.train_size)], "7. Run Diagnostic Analysis (Memorisation, Similarity, Worst Cases, Per-Protein)")

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETE")
    print(f"Total pipeline elapsed time: {total_time:.2f} seconds ({total_time / 60:.1f} minutes)")
    print("All artifacts generated and validated:")
    print("  - Results Table:     results/four_regime_table.csv")
    print("  - Regime Chart:      results/four_regime_chart.png")
    print("  - Memorisation:      results/memorisation.csv")
    print("  - Similarity Curve:  results/auc_vs_similarity.png")
    print("  - Worst Errors:      results/worst_predictions.csv & worst_false_positives.csv")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
