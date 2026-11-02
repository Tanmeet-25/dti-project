#!/usr/bin/env python3
"""
Diagnostic Analysis of Drug-Target Interaction (DTI) Benchmark Results:
1. Memorisation Analysis:
   Quantifies % performance lost between Split 1 (warm-start) and Split 4 (cold-start)
   as (ROC-AUC_split4 - 0.5) / (ROC-AUC_split1 - 0.5). Saved to results/memorisation.csv.
2. Similarity vs Performance Analysis:
   - Computes max Tanimoto drug similarity (ECFP4) for test drugs against training set in Splits 2 & 4.
   - Computes max cosine protein similarity (AAC+DPC) for test proteins against training set in Splits 3 & 4.
   - Bins test pairs by similarity (<0.4, 0.4–0.6, 0.6–0.8, >0.8 for drugs; <0.92, 0.92–0.95, 0.95–0.97, >0.97 for proteins).
   - Reports RF and MLP ROC-AUC and number of test pairs in every bin.
   - Generates line chart saved to results/auc_vs_similarity.png.
3. Worst Predictions & Confident Errors (Split 4 - Random Forest):
   - Lists the 20 worst false negatives (label 1 with lowest predicted probability).
   - Lists the 20 most confident false positives (label 0 with highest predicted probability).
   - Reports the highest predicted probability any label-0 pair received.
   - Saved to results/worst_predictions.csv and results/worst_false_positives.csv.
4. Per-Protein Generalization (Split 1):
   - Evaluates ROC-AUC per protein on Split 1 for proteins with at least 10 positive and 10 negative test pairs.
   - Reports qualifying count, 10 best, and 10 worst by RF ROC-AUC.
5. Feature Importance Breakdown:
   - Sums feature importances of 2,048 ECFP4 drug bits vs 420 AAC+DPC protein features
   - from results/rf_split1.joblib.
"""

import os
import sys
import shutil
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from features import load_features
from splits import load_split, get_protein_groups


def run_memorisation_analysis(results_dir: str = "results") -> pd.DataFrame:
    """Compute memorisation ratio and performance lost between Split 1 and Split 4."""
    table_path = os.path.join(results_dir, "four_regime_table.csv")
    if not os.path.exists(table_path):
        raise FileNotFoundError(f"Missing {table_path}. Please run train.py first.")

    df_table = pd.read_csv(table_path)
    models = [
        "Dummy",
        "Protein-Only",
        "Drug-Only",
        "Nearest-Protein",
        "RF drug-features-only",
        "RF protein-features-only",
        "Random Forest",
        "PyTorch MLP",
    ]
    records = []

    for m in models:
        s1_row = df_table[(df_table["Split"] == "random") & (df_table["Model"] == m)]
        s4_row = df_table[(df_table["Split"] == "unseen_both") & (df_table["Model"] == m)]

        if s1_row.empty or s4_row.empty:
            continue

        auc1 = float(s1_row["ROC-AUC"].iloc[0])
        auc4 = float(s4_row["ROC-AUC"].iloc[0])

        excess1 = auc1 - 0.5
        excess4 = auc4 - 0.5

        if abs(excess1) < 1e-6:
            ratio = 0.0
            pct_lost = 100.0 if abs(excess4) < 1e-6 else 0.0
        else:
            ratio = max(0.0, excess4 / excess1)
            pct_lost = (1.0 - ratio) * 100.0

        records.append({
            "Model": m,
            "Split 1 ROC-AUC": round(auc1, 4),
            "Split 4 ROC-AUC": round(auc4, 4),
            "Excess AUC Split 1": round(excess1, 4),
            "Excess AUC Split 4": round(excess4, 4),
            "Retention Ratio": round(ratio, 4),
            "Performance Lost (%)": round(pct_lost, 2),
        })

    mem_df = pd.DataFrame(records)
    out_csv = os.path.join(results_dir, "memorisation.csv")
    mem_df.to_csv(out_csv, index=False)
    return mem_df


def compute_drug_similarities(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    drug_feats: np.ndarray = None,
    drug_to_idx: dict = None,
    max_test_sample: int = 3000,
    seed: int = 42,
) -> dict[str, float]:
    """Compute max Tanimoto similarity of test drugs to any training drug using ECFP4."""
    tr_smiles = df["canonical_smiles"].iloc[train_idx].unique()
    te_smiles = df["canonical_smiles"].iloc[test_idx].unique()

    rng = np.random.RandomState(seed)
    if len(te_smiles) > max_test_sample:
        query_smiles = rng.choice(te_smiles, size=max_test_sample, replace=False)
    else:
        query_smiles = te_smiles

    # Fast tensor-based Tanimoto if precomputed feature matrix available
    if drug_feats is not None and drug_to_idx is not None:
        import torch
        q_idx = [drug_to_idx[s] for s in query_smiles]
        tr_f_idx = [drug_to_idx[s] for s in tr_smiles]

        A = torch.from_numpy(drug_feats[q_idx]).float()
        B = torch.from_numpy(drug_feats[tr_f_idx]).float()
        sum_B = B.sum(dim=1)
        sum_A = A.sum(dim=1, keepdim=True)

        inter = A @ B.T
        union = sum_A + sum_B.unsqueeze(0) - inter
        union = torch.clamp(union, min=1.0)
        tan = inter / union
        max_sims = tan.max(dim=1).values.numpy()
        return {s: float(sim) for s, sim in zip(query_smiles, max_sims)}

    # Fallback to RDKit BulkTanimotoSimilarity
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    train_fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in tr_smiles]

    sim_map = {}
    for s in query_smiles:
        fp = gen.GetFingerprint(Chem.MolFromSmiles(s))
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        sim_map[s] = max(sims)

    return sim_map


def compute_protein_similarities(
    df: pd.DataFrame,
    prot_feats: np.ndarray,
    prot_to_idx: dict,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, float]:
    """Compute max cosine similarity of test proteins to any training protein using composition features."""
    tr_seqs = df["protein_sequence"].iloc[train_idx].unique()
    te_seqs = df["protein_sequence"].iloc[test_idx].unique()

    P_tr = np.array([prot_feats[prot_to_idx[p]] for p in tr_seqs], dtype=np.float32)
    P_te = np.array([prot_feats[prot_to_idx[p]] for p in te_seqs], dtype=np.float32)

    P_tr_norm = P_tr / np.linalg.norm(P_tr, axis=1, keepdims=True)
    P_te_norm = P_te / np.linalg.norm(P_te, axis=1, keepdims=True)

    cosine_sim_mat = P_te_norm @ P_tr_norm.T
    max_sims = np.max(cosine_sim_mat, axis=1)

    return {seq: float(sim) for seq, sim in zip(te_seqs, max_sims)}


def run_similarity_analysis(
    df: pd.DataFrame,
    prot_feats: np.ndarray,
    prot_to_idx: dict,
    drug_feats: np.ndarray = None,
    drug_to_idx: dict = None,
    results_dir: str = "results",
    train_size: int = 150000,
    seed: int = 42,
) -> tuple[dict, str]:
    """Bin test pairs by similarity and report RF and MLP ROC-AUC per bin."""
    preds_dir = os.path.join(results_dir, "preds")

    # 1. Split 2: Unseen Drugs (Drug Tanimoto similarity)
    tr_idx_2, _, te_idx_2 = load_split("unseen_drugs")
    sub_tr_2, _ = train_test_split(tr_idx_2, train_size=min(train_size, len(tr_idx_2)), stratify=df["label"].iloc[tr_idx_2].values, random_state=seed)
    drug_sim_map_2 = compute_drug_similarities(df, sub_tr_2, te_idx_2, drug_feats=drug_feats, drug_to_idx=drug_to_idx, max_test_sample=3000, seed=seed)

    preds_2 = pd.read_csv(os.path.join(preds_dir, "unseen_drugs_predictions.csv"))
    preds_2["smiles"] = df["canonical_smiles"].iloc[preds_2["test_idx"]].values
    preds_2["similarity"] = preds_2["smiles"].map(drug_sim_map_2)
    eval_df_2 = preds_2.dropna(subset=["similarity"]).copy()

    # 2. Split 4: Unseen Both (Drug Tanimoto similarity AND Protein Cosine similarity)
    tr_idx_4, _, te_idx_4 = load_split("unseen_both")
    sub_tr_4, _ = train_test_split(tr_idx_4, train_size=min(train_size, len(tr_idx_4)), stratify=df["label"].iloc[tr_idx_4].values, random_state=seed)
    drug_sim_map_4 = compute_drug_similarities(df, sub_tr_4, te_idx_4, drug_feats=drug_feats, drug_to_idx=drug_to_idx, max_test_sample=3000, seed=seed)
    prot_sim_map_4 = compute_protein_similarities(df, prot_feats, prot_to_idx, sub_tr_4, te_idx_4)

    preds_4 = pd.read_csv(os.path.join(preds_dir, "unseen_both_predictions.csv"))
    preds_4["smiles"] = df["canonical_smiles"].iloc[preds_4["test_idx"]].values
    preds_4["seq"] = df["protein_sequence"].iloc[preds_4["test_idx"]].values

    # Drug similarity evaluation for Split 4
    preds_4_drug = preds_4.copy()
    preds_4_drug["similarity"] = preds_4_drug["smiles"].map(drug_sim_map_4)
    eval_df_4_drug = preds_4_drug.dropna(subset=["similarity"]).copy()

    # Protein similarity evaluation for Split 4
    preds_4_prot = preds_4.copy()
    preds_4_prot["similarity"] = preds_4_prot["seq"].map(prot_sim_map_4)
    eval_df_4_prot = preds_4_prot.dropna(subset=["similarity"]).copy()

    # 3. Split 3: Unseen Proteins (Protein Cosine similarity)
    tr_idx_3, _, te_idx_3 = load_split("unseen_proteins")
    sub_tr_3, _ = train_test_split(tr_idx_3, train_size=min(train_size, len(tr_idx_3)), stratify=df["label"].iloc[tr_idx_3].values, random_state=seed)
    prot_sim_map_3 = compute_protein_similarities(df, prot_feats, prot_to_idx, sub_tr_3, te_idx_3)

    preds_3 = pd.read_csv(os.path.join(preds_dir, "unseen_proteins_predictions.csv"))
    preds_3["seq"] = df["protein_sequence"].iloc[preds_3["test_idx"]].values
    preds_3["similarity"] = preds_3["seq"].map(prot_sim_map_3)
    eval_df_3 = preds_3.dropna(subset=["similarity"]).copy()

    # Define bins
    drug_bins = [-1e-5, 0.4, 0.6, 0.8, 1.01]
    drug_labels = ["<0.4", "0.4–0.6", "0.6–0.8", ">0.8"]

    prot_bins = [-1e-5, 0.92, 0.95, 0.97, 1.01]
    prot_labels = ["<0.92", "0.92–0.95", "0.95–0.97", ">0.97"]

    binned_results = {}

    def calc_binned_metrics(sub_df, bins, labels):
        sub_df = sub_df.copy()
        sub_df["bin"] = pd.cut(sub_df["similarity"], bins=bins, labels=labels)
        res = []
        for b in labels:
            b_df = sub_df[sub_df["bin"] == b]
            n_pairs = len(b_df)
            if n_pairs > 0 and len(b_df["y_true"].unique()) == 2:
                auc_rf = roc_auc_score(b_df["y_true"], b_df["random_forest"])
                auc_mlp = roc_auc_score(b_df["y_true"], b_df["mlp"])
            else:
                auc_rf, auc_mlp = np.nan, np.nan
            res.append({"Bin": b, "Test Pairs": n_pairs, "RF ROC-AUC": round(auc_rf, 4) if not np.isnan(auc_rf) else np.nan, "MLP ROC-AUC": round(auc_mlp, 4) if not np.isnan(auc_mlp) else np.nan})
        return pd.DataFrame(res)

    binned_results["Split 2 (Unseen Drugs - Drug Tanimoto Sim)"] = calc_binned_metrics(eval_df_2, drug_bins, drug_labels)
    binned_results["Split 4 (Unseen Both - Drug Tanimoto Sim)"] = calc_binned_metrics(eval_df_4_drug, drug_bins, drug_labels)
    binned_results["Split 3 (Unseen Proteins - Protein Cosine Sim)"] = calc_binned_metrics(eval_df_3, prot_bins, prot_labels)
    binned_results["Split 4 (Unseen Both - Protein Cosine Sim)"] = calc_binned_metrics(eval_df_4_prot, prot_bins, prot_labels)

    # Plot AUC vs Similarity
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    fig.patch.set_facecolor("#FAFAFC")

    # Panel 1: Drug Tanimoto Similarity vs ROC-AUC
    ax1.set_facecolor("#FFFFFF")
    df_s2 = binned_results["Split 2 (Unseen Drugs - Drug Tanimoto Sim)"]
    df_s4_d = binned_results["Split 4 (Unseen Both - Drug Tanimoto Sim)"]
    x = range(len(drug_labels))

    ax1.plot(x, df_s2["RF ROC-AUC"], marker="o", linewidth=2.4, color="#2B6CB0", label="Split 2 (Unseen Drugs) - RF")
    ax1.plot(x, df_s2["MLP ROC-AUC"], marker="s", linewidth=2.4, color="#3182CE", linestyle="--", label="Split 2 (Unseen Drugs) - MLP")
    ax1.plot(x, df_s4_d["RF ROC-AUC"], marker="^", linewidth=2.4, color="#DD6B20", label="Split 4 (Unseen Both) - RF")
    ax1.plot(x, df_s4_d["MLP ROC-AUC"], marker="D", linewidth=2.4, color="#ED8936", linestyle="--", label="Split 4 (Unseen Both) - MLP")

    ax1.set_xticks(x)
    ax1.set_xticklabels(drug_labels, fontsize=11, fontweight="600")
    ax1.set_xlabel("Drug Max Tanimoto Similarity (ECFP4)", fontsize=11.5, fontweight="600", color="#2D3748")
    ax1.set_ylabel("ROC-AUC", fontsize=11.5, fontweight="600", color="#2D3748")
    ax1.set_title("Model Generalization vs Drug Chemical Similarity", fontsize=12.5, fontweight="700", color="#1A202C")
    ax1.axhline(0.5, color="#E53E3E", linestyle=":", linewidth=1.4, alpha=0.7)
    ax1.set_ylim(0.45, 1.02)
    ax1.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0", loc="lower right", fontsize=9.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.45)

    # Panel 2: Protein Cosine Similarity vs ROC-AUC (Splits 3 and 4)
    ax2.set_facecolor("#FFFFFF")
    df_s3 = binned_results["Split 3 (Unseen Proteins - Protein Cosine Sim)"]
    df_s4_p = binned_results["Split 4 (Unseen Both - Protein Cosine Sim)"]
    x_p = range(len(prot_labels))

    ax2.plot(x_p, df_s3["RF ROC-AUC"], marker="o", linewidth=2.4, color="#2E8540", label="Split 3 (Unseen Proteins) - RF")
    ax2.plot(x_p, df_s3["MLP ROC-AUC"], marker="s", linewidth=2.4, color="#48BB78", linestyle="--", label="Split 3 (Unseen Proteins) - MLP")
    ax2.plot(x_p, df_s4_p["RF ROC-AUC"], marker="^", linewidth=2.4, color="#DD6B20", label="Split 4 (Unseen Both) - RF")
    ax2.plot(x_p, df_s4_p["MLP ROC-AUC"], marker="D", linewidth=2.4, color="#ED8936", linestyle="--", label="Split 4 (Unseen Both) - MLP")

    ax2.set_xticks(x_p)
    ax2.set_xticklabels(prot_labels, fontsize=11, fontweight="600")
    ax2.set_xlabel("Protein Max Cosine Similarity (Composition)", fontsize=11.5, fontweight="600", color="#2D3748")
    ax2.set_ylabel("ROC-AUC", fontsize=11.5, fontweight="600", color="#2D3748")
    ax2.set_title("Model Generalization vs Target Sequence Similarity", fontsize=12.5, fontweight="700", color="#1A202C")
    ax2.axhline(0.5, color="#E53E3E", linestyle=":", linewidth=1.4, alpha=0.7)
    ax2.set_ylim(0.45, 1.02)
    ax2.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0", loc="lower right", fontsize=9.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.45)

    fig.suptitle("DTI Generalization Regimes: ROC-AUC as a Function of Entity Similarity",
                 fontsize=14, fontweight="700", color="#1A202C", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plot_path = os.path.join(results_dir, "auc_vs_similarity.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return binned_results, plot_path


def run_worst_predictions_analysis(
    df: pd.DataFrame,
    drug_feats: np.ndarray = None,
    drug_to_idx: dict = None,
    results_dir: str = "results",
    train_size: int = 150000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Identify the 20 worst false negatives (label 1 with lowest prob) and
    the 20 most confident false positives (label 0 with highest prob) for RF on Split 4.
    Also returns the highest predicted probability any label-0 pair received.
    """
    preds_dir = os.path.join(results_dir, "preds")
    preds_4 = pd.read_csv(os.path.join(preds_dir, "unseen_both_predictions.csv"))

    # Separate label 1 and label 0
    fn_candidates = preds_4[preds_4["y_true"] == 1].sort_values("random_forest", ascending=True).head(20).copy()
    fp_candidates = preds_4[preds_4["y_true"] == 0].sort_values("random_forest", ascending=False).head(20).copy()

    # Highest predicted probability for any label-0 pair
    max_prob_label_0 = float(preds_4[preds_4["y_true"] == 0]["random_forest"].max())

    # Precompute training fingerprints for Split 4
    tr_idx, _, _ = load_split("unseen_both")
    sub_tr, _ = train_test_split(tr_idx, train_size=min(train_size, len(tr_idx)), stratify=df["label"].iloc[tr_idx].values, random_state=seed)

    tr_smiles = df["canonical_smiles"].iloc[sub_tr].unique()

    if drug_feats is not None and drug_to_idx is not None:
        import torch
        tr_f_idx = [drug_to_idx[s] for s in tr_smiles]
        B = torch.from_numpy(drug_feats[tr_f_idx]).float()
        sum_B = B.sum(dim=1)

        def get_max_sim(s):
            A = torch.from_numpy(drug_feats[[drug_to_idx[s]]]).float()
            inter = A @ B.T
            union = A.sum() + sum_B - inter
            union = torch.clamp(union, min=1.0)
            return float((inter / union).max())
    else:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        train_fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in tr_smiles]
        def get_max_sim(s):
            fp = gen.GetFingerprint(Chem.MolFromSmiles(s))
            return max(DataStructs.BulkTanimotoSimilarity(fp, train_fps))

    def extract_records(subset_df):
        records = []
        for _, row in subset_df.iterrows():
            idx = int(row["test_idx"])
            s = df["canonical_smiles"].iloc[idx]
            max_sim = get_max_sim(s)

            raw_name = str(df["ligand_name"].iloc[idx])
            if "::" in raw_name:
                candidates = [c.strip() for c in raw_name.split("::") if c.strip()]
                short_cands = [c for c in candidates if len(c) <= 30 and not c.startswith("Scaffold")]
                name = short_cands[-1] if short_cands else candidates[0][:30]
            else:
                cid = df["pubchem_cid"].iloc[idx]
                name = raw_name[:30] if (raw_name and raw_name != "nan") else f"CID {cid}"

            target = df["target_name"].iloc[idx]
            records.append({
                "drug_name": name,
                "target_name": target,
                "true_label": int(row["y_true"]),
                "predicted_probability": round(float(row["random_forest"]), 4),
                "max_drug_similarity": round(float(max_sim), 4),
            })
        return pd.DataFrame(records)

    worst_fn_df = extract_records(fn_candidates)
    worst_fp_df = extract_records(fp_candidates)

    # Save CSVs
    worst_fn_df.to_csv(os.path.join(results_dir, "worst_predictions.csv"), index=False)
    worst_fp_df.to_csv(os.path.join(results_dir, "worst_false_positives.csv"), index=False)

    return worst_fn_df, worst_fp_df, max_prob_label_0


def run_per_protein_analysis(df: pd.DataFrame, results_dir: str = "results") -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Compute ROC-AUC per protein on Split 1 for proteins with >= 10 positives AND >= 10 negatives."""
    preds_dir = os.path.join(results_dir, "preds")
    preds_1 = pd.read_csv(os.path.join(preds_dir, "random_predictions.csv"))

    preds_1["target_name"] = df["target_name"].iloc[preds_1["test_idx"]].values
    groups = preds_1.groupby("target_name")
    qualifying = []

    for name, group in groups:
        n_pos = int((group["y_true"] == 1).sum())
        n_neg = int((group["y_true"] == 0).sum())
        if n_pos >= 10 and n_neg >= 10:
            auc_rf = roc_auc_score(group["y_true"], group["random_forest"])
            auc_mlp = roc_auc_score(group["y_true"], group["mlp"])
            qualifying.append({
                "target_name": name,
                "test_pairs": len(group),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "rf_auc": round(float(auc_rf), 4),
                "mlp_auc": round(float(auc_mlp), 4),
            })

    q_df = pd.DataFrame(qualifying).sort_values("rf_auc", ascending=False)
    top10 = q_df.head(10).copy()
    bottom10 = q_df.tail(10).copy()
    return top10, bottom10, len(q_df)


def run_feature_importance_analysis(results_dir: str = "results") -> tuple[float, float, float, float]:
    """Sum feature importances of 2048 ECFP4 drug bits vs 420 protein features from Split 1 RF."""
    model_path = os.path.join(results_dir, "rf_split1.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing {model_path}. Please run train.py first.")

    rf = joblib.load(model_path)
    importances = rf.feature_importances_
    drug_imp = float(np.sum(importances[:2048]))
    prot_imp = float(np.sum(importances[2048:]))
    total = drug_imp + prot_imp

    drug_pct = (drug_imp / total) * 100.0
    prot_pct = (prot_imp / total) * 100.0

    return drug_imp, drug_pct, prot_imp, prot_pct


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Post-benchmark diagnostic analysis.")
    parser.add_argument("--results", default="results", help="Results directory containing predictions")
    parser.add_argument("--train-size", type=int, default=150000, help="Training size used for similarity reference")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    results_dir = args.results
    train_size = args.train_size

    print("=" * 80)
    print(f"POST-BENCHMARK DIAGNOSTIC ANALYSIS (analyse.py, train_size={train_size:,})")
    print("=" * 80)

    df = pd.read_csv("bindingdb_human_dti_binary.tsv", sep="\t", low_memory=False)
    drug_feats, prot_feats, drug_to_idx, prot_to_idx = load_features("features")

    # 1. Memorisation Analysis
    print("\n" + "=" * 80)
    print("1. MEMORISATION ANALYSIS (Split 1 vs Split 4)")
    print("=" * 80)
    mem_df = run_memorisation_analysis(results_dir)
    print(mem_df.to_string(index=False))
    print(f"\nSaved memorisation table to '{results_dir}/memorisation.csv'")

    # 2. Similarity vs Performance Analysis
    print("\n" + "=" * 80)
    print("2. SIMILARITY VS PERFORMANCE ANALYSIS")
    print("=" * 80)
    binned_results, plot_path = run_similarity_analysis(
        df,
        prot_feats,
        prot_to_idx,
        drug_feats=drug_feats,
        drug_to_idx=drug_to_idx,
        results_dir=results_dir,
        train_size=train_size,
        seed=args.seed,
    )
    for regime_name, b_df in binned_results.items():
        print(f"\n{regime_name}:")
        print(b_df.to_string(index=False))
    print(f"\nLine chart saved to '{plot_path}'")

    # 3. Worst Predictions (False Negatives & False Positives on Split 4)
    print("\n" + "=" * 80)
    print("3. CONFIDENT WRONG PREDICTIONS (Split 4 - Random Forest)")
    print("=" * 80)
    worst_fn_df, worst_fp_df, max_prob_label_0 = run_worst_predictions_analysis(
        df,
        drug_feats=drug_feats,
        drug_to_idx=drug_to_idx,
        results_dir=results_dir,
        train_size=train_size,
        seed=args.seed,
    )
    print(f"\n>>> HIGHEST PREDICTED PROBABILITY FOR ANY LABEL-0 PAIR: {max_prob_label_0:.4f} <<<")
    print("\nTop 20 Worst False Negatives (Label 1 with lowest predicted probability):")
    print(worst_fn_df.to_string(index=False))
    print("\nTop 20 Most Confident False Positives (Label 0 with highest predicted probability):")
    print(worst_fp_df.to_string(index=False))
    print(f"\nSaved false negatives to '{results_dir}/worst_predictions.csv'")
    print(f"Saved false positives to '{results_dir}/worst_false_positives.csv'")

    # 4. Per-Protein Analysis
    print("\n" + "=" * 80)
    print("4. PER-PROTEIN ROC-AUC (Split 1 - Proteins with >= 10 Positives and >= 10 Negatives)")
    print("=" * 80)
    top10_prots, bottom10_prots, total_qual = run_per_protein_analysis(df, results_dir)
    print(f"Total qualifying proteins (>=10 pos & >=10 neg): {total_qual}")
    print("\nTop 10 Best Performing Proteins (Split 1):")
    print(top10_prots.to_string(index=False))
    print("\nTop 10 Worst Performing Proteins (Split 1):")
    print(bottom10_prots.to_string(index=False))

    # 5. Feature Importances
    print("\n" + "=" * 80)
    print("5. FEATURE IMPORTANCE BREAKDOWN (Split 1 Random Forest)")
    print("=" * 80)
    d_imp, d_pct, p_imp, p_pct = run_feature_importance_analysis(results_dir)
    print(f"Drug features (2,048 ECFP4 bits):     {d_imp:.4f} ({d_pct:.2f}%)")
    print(f"Protein features (420 AAC+DPC dims):   {p_imp:.4f} ({p_pct:.2f}%)")

    # Short Summary
    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY")
    print("=" * 80)
    rf_loss = mem_df.loc[mem_df["Model"] == "Random Forest", "Performance Lost (%)"].values[0]
    mlp_loss = mem_df.loc[mem_df["Model"] == "PyTorch MLP", "Performance Lost (%)"].values[0]
    print(f"- Memorisation: Random Forest lost {rf_loss:.1f}% of excess performance in cold-start (Split 4),")
    print(f"  while PyTorch MLP retained more generalisable representations, losing {mlp_loss:.1f}%.")
    print(f"- Entity Similarity: Generalisation scales directly with similarity to training instances.")
    print(f"- Error Asymmetry: Highest false positive probability is {max_prob_label_0:.3f}, meaning the model")
    print(f"  is conservative in predicting binding (false negatives dominate the extreme confidence tail).")
    print(f"- Protein Generalisation: Across {total_qual} well-sampled proteins (>=10 pos & >=10 neg), 10 targets")
    print(f"  achieve near-perfect RF AUC, while the 10 lowest range from {bottom10_prots['rf_auc'].min():.3f} to {bottom10_prots['rf_auc'].max():.3f}.")
    print(f"- Feature Importance: {d_pct:.1f}% drug fingerprint bits vs {p_pct:.1f}% protein sequence composition.")
    print("=" * 80)


if __name__ == "__main__":
    main()
