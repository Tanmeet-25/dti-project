#!/usr/bin/env python3
"""
Train and evaluate DTI benchmark models across four splitting regimes:
(a) Dummy baseline: always predicts training positive rate.
(b) Protein-only baseline: predicts each protein's positive rate in training.
(c) Drug-only baseline: predicts each drug's positive rate in training.
(d) Nearest-protein baseline: finds most similar training protein by cosine similarity
    of composition features (excluding same protein) and predicts its positive rate.
(e) RF drug-features-only: 200 trees on 2,048 ECFP4 fingerprint bits only.
(f) RF protein-features-only: 200 trees on 420 AAC+DPC composition features only.
(g) RandomForestClassifier: 200 trees on concatenated 2,468 features.
(h) PyTorch MLP: 2468 -> 512 -> 128 -> 1, ReLU, dropout 0.3, Adam lr 1e-3, batch 512,
    weighted BCE loss, early stopping on validation ROC-AUC.

CLI Options:
- `--train-size`: Uniform training sample size per split (default: 40000).
- `--seed`: Random seed (default: 42).

Outputs:
- results/four_regime_table.csv (ROC-AUC & PR-AUC with 1000-sample bootstrap 95% CIs)
- results/four_regime_chart.png (grouped bar chart across regimes)
- results/preds/{split}_{model}.npz & {split}_predictions.csv (test predictions)
- results/rf_split1.joblib (compressed Random Forest model for Split 1 only)
"""

import os
import sys
import copy
import time
import json
import base64
import shutil
import psutil
import joblib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from features import load_features
from splits import load_split, get_protein_groups


# -----------------------------------------------------------------------------
# PyTorch MLP Architecture
# -----------------------------------------------------------------------------
class DTIMLP(nn.Module):
    def __init__(self, in_dim: int = 2468, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# -----------------------------------------------------------------------------
# Batch Feature Construction
# -----------------------------------------------------------------------------
def build_pair_features(
    indices: np.ndarray,
    df: pd.DataFrame,
    drug_features: np.ndarray,
    protein_features: np.ndarray,
    drug_to_idx: dict,
    prot_to_idx: dict,
    batch_size: int = 20000,
) -> np.ndarray:
    """
    Build concatenated drug + protein features (2048 + 420 = 2468 float32)
    in batches to avoid memory spikes.
    """
    n_samples = len(indices)
    total_dim = drug_features.shape[1] + protein_features.shape[1]
    X = np.empty((n_samples, total_dim), dtype=np.float32)

    smiles_col = df["canonical_smiles"].iloc[indices].values
    seq_col = df["protein_sequence"].iloc[indices].values

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_smiles = smiles_col[start:end]
        batch_seqs = seq_col[start:end]

        d_idx = [drug_to_idx[s] for s in batch_smiles]
        p_idx = [prot_to_idx[s] for s in batch_seqs]

        batch_drugs = drug_features[d_idx].astype(np.float32)
        batch_prots = protein_features[p_idx]
        X[start:end, : drug_features.shape[1]] = batch_drugs
        X[start:end, drug_features.shape[1] :] = batch_prots

    return X


# -----------------------------------------------------------------------------
# Fast Bootstrap 95% Confidence Intervals
# -----------------------------------------------------------------------------
def compute_metrics_and_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, tuple[float, float], float, tuple[float, float]]:
    """
    Compute ROC-AUC and PR-AUC point estimates along with 1000-sample bootstrap 95% CIs.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)

    # Point estimates
    is_constant = len(np.unique(y_pred)) <= 1
    if is_constant:
        roc_pt = 0.5
        pr_pt = float(np.mean(y_true))
    else:
        try:
            roc_pt = float(roc_auc_score(y_true, y_pred))
        except Exception:
            roc_pt = 0.5
        try:
            pr_pt = float(average_precision_score(y_true, y_pred))
        except Exception:
            pr_pt = float(np.mean(y_true))

    # Bootstrap resampling
    rng = np.random.RandomState(seed)
    roc_boots = []
    pr_boots = []

    if is_constant:
        for _ in range(n_boot):
            b_idx = rng.randint(0, n, size=n)
            roc_boots.append(0.5)
            pr_boots.append(float(np.mean(y_true[b_idx])))
    else:
        for _ in range(n_boot):
            b_idx = rng.randint(0, n, size=n)
            yt_b = y_true[b_idx]
            yp_b = y_pred[b_idx]
            if len(np.unique(yt_b)) < 2:
                continue
            try:
                roc_boots.append(float(roc_auc_score(yt_b, yp_b)))
            except Exception:
                roc_boots.append(0.5)
            try:
                pr_boots.append(float(average_precision_score(yt_b, yp_b)))
            except Exception:
                pr_boots.append(float(np.mean(yt_b)))

    roc_ci = (float(np.percentile(roc_boots, 2.5)), float(np.percentile(roc_boots, 97.5)))
    pr_ci = (float(np.percentile(pr_boots, 2.5)), float(np.percentile(pr_boots, 97.5)))

    return roc_pt, roc_ci, pr_pt, pr_ci


# -----------------------------------------------------------------------------
# Baseline Models
# -----------------------------------------------------------------------------
def run_dummy_baseline(y_train: np.ndarray, n_test: int) -> np.ndarray:
    """Predict training positive rate constantly."""
    p_train = float(np.mean(y_train))
    return np.full(n_test, p_train, dtype=float)


def run_protein_only_baseline(
    df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray
) -> np.ndarray:
    """Predict each protein's positive rate in training (global rate if unseen)."""
    prot_groups = get_protein_groups(df).values
    train_prots = prot_groups[train_idx]
    train_labels = df["label"].iloc[train_idx].values

    global_rate = float(np.mean(train_labels))
    train_df = pd.DataFrame({"protein": train_prots, "label": train_labels})
    prot_rates = train_df.groupby("protein")["label"].mean().to_dict()

    test_prots = prot_groups[test_idx]
    preds = np.array([prot_rates.get(p, global_rate) for p in test_prots], dtype=float)
    return preds


def run_drug_only_baseline(
    df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray
) -> np.ndarray:
    """Predict each drug's positive rate in training (global rate if unseen)."""
    drugs = df["canonical_smiles"].values
    train_drugs = drugs[train_idx]
    train_labels = df["label"].iloc[train_idx].values

    global_rate = float(np.mean(train_labels))
    train_df = pd.DataFrame({"drug": train_drugs, "label": train_labels})
    drug_rates = train_df.groupby("drug")["label"].mean().to_dict()

    test_drugs = drugs[test_idx]
    preds = np.array([drug_rates.get(d, global_rate) for d in test_drugs], dtype=float)
    return preds


def run_nearest_protein_baseline(
    df: pd.DataFrame,
    protein_features: np.ndarray,
    prot_to_idx: dict,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> np.ndarray:
    """
    Nearest-protein baseline: for each test pair, find the most similar training protein
    by cosine similarity of composition features (excluding the same protein sequence and UniProt ID)
    and predict that nearest protein's positive rate in training.
    """
    prot_groups = get_protein_groups(df).values
    seqs = df["protein_sequence"].values
    labels = df["label"].values

    train_seqs_all = seqs[train_idx]
    train_groups_all = prot_groups[train_idx]
    train_labels_all = labels[train_idx]

    global_rate = float(np.mean(train_labels_all))

    train_df = pd.DataFrame({
        "seq": train_seqs_all,
        "group": train_groups_all,
        "label": train_labels_all,
    })
    prot_rates = train_df.groupby("seq")["label"].mean().to_dict()
    seq_to_group = train_df.groupby("seq")["group"].first().to_dict()

    tr_unique_seqs = np.array(list(prot_rates.keys()))
    tr_unique_groups = np.array([seq_to_group[s] for s in tr_unique_seqs])

    P_tr = np.array([protein_features[prot_to_idx[s]] for s in tr_unique_seqs], dtype=np.float32)
    P_tr_norm = P_tr / np.linalg.norm(P_tr, axis=1, keepdims=True)

    test_seqs_all = seqs[test_idx]
    te_unique_seqs = np.unique(test_seqs_all)
    test_df = pd.DataFrame({"seq": seqs[test_idx], "group": prot_groups[test_idx]})
    te_seq_to_group = test_df.groupby("seq")["group"].first().to_dict()

    P_te = np.array([protein_features[prot_to_idx[s]] for s in te_unique_seqs], dtype=np.float32)
    P_te_norm = P_te / np.linalg.norm(P_te, axis=1, keepdims=True)

    cosine_sims = P_te_norm @ P_tr_norm.T

    nearest_rate_map = {}
    for i, s_te in enumerate(te_unique_seqs):
        grp_te = te_seq_to_group[s_te]
        valid_mask = (tr_unique_seqs != s_te) & (tr_unique_groups != grp_te)
        if np.any(valid_mask):
            masked_sims = np.where(valid_mask, cosine_sims[i], -999.0)
            best_j = np.argmax(masked_sims)
            nearest_rate_map[s_te] = prot_rates[tr_unique_seqs[best_j]]
        else:
            nearest_rate_map[s_te] = global_rate

    preds = np.array([nearest_rate_map.get(s, global_rate) for s in test_seqs_all], dtype=float)
    return preds


# -----------------------------------------------------------------------------
# Random Forest Model
# -----------------------------------------------------------------------------
def run_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, RandomForestClassifier, float]:
    """Train RandomForestClassifier(200 trees, sqrt features, balanced)."""
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=200,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(X_train, y_train)
    t_train = time.time() - t0
    preds = rf.predict_proba(X_test)[:, 1]
    return preds, rf, t_train


# -----------------------------------------------------------------------------
# PyTorch MLP Model
# -----------------------------------------------------------------------------
def run_pytorch_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    seed: int = 42,
    max_epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 4,
    return_model: bool = False,
) -> tuple:
    """Train PyTorch MLP with weighted loss and early stopping on validation ROC-AUC."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    # Compute positive class weight: neg / pos
    n_pos = float(np.sum(y_train == 1))
    n_neg = float(np.sum(y_train == 0))
    pos_weight = n_neg / max(n_pos, 1.0)
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)

    # Prepare datasets and loaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).float())
    test_ds = TensorDataset(torch.from_numpy(X_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = DTIMLP(in_dim=X_train.shape[1], dropout=0.3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_auc = -1.0
    best_weights = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        # Validation evaluation
        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for bx, _ in val_loader:
                probs = torch.sigmoid(model(bx))
                val_preds_list.append(probs.numpy())
        val_preds = np.concatenate(val_preds_list)

        try:
            val_auc = roc_auc_score(y_val, val_preds)
        except Exception:
            val_auc = 0.5

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    t_train = time.time() - t0

    # Test prediction
    model.eval()
    test_preds_list = []
    with torch.no_grad():
        for bx, in test_loader:
            probs = torch.sigmoid(model(bx))
            test_preds_list.append(probs.numpy())
    test_preds = np.concatenate(test_preds_list)

    if return_model:
        return test_preds, t_train, model
    return test_preds, t_train


# -----------------------------------------------------------------------------
# Grouped Bar Chart Plotting
# -----------------------------------------------------------------------------
def plot_four_regime_chart(
    results_df: pd.DataFrame, output_path: str = "results/four_regime_chart.png"
):
    """Generate high-resolution grouped bar chart comparing models across 4 splits."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(19, 7.5), dpi=300)
    fig.patch.set_facecolor("#FAFAFC")

    splits = ["random", "unseen_drugs", "unseen_proteins", "unseen_both"]
    split_labels = ["1. Random Pairs", "2. Unseen Drugs", "3. Unseen Proteins", "4. Unseen Both"]
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
    model_colors = [
        "#A0AEC0",  # Dummy (gray)
        "#63B3ED",  # Protein-Only (light blue)
        "#F6AD55",  # Drug-Only (light orange)
        "#D69E2E",  # Nearest-Protein (gold/yellow)
        "#DD6B20",  # RF drug-features-only (dark orange)
        "#3182CE",  # RF protein-features-only (blue)
        "#38A169",  # Random Forest (green)
        "#805AD5",  # PyTorch MLP (purple)
    ]

    n_splits = len(splits)
    n_models = len(models)
    width = 0.095
    indices = np.arange(n_splits)

    for i, (m, col) in enumerate(zip(models, model_colors)):
        roc_vals = []
        roc_errs_lo = []
        roc_errs_hi = []
        pr_vals = []
        pr_errs_lo = []
        pr_errs_hi = []

        for s in splits:
            sub = results_df[(results_df["Split"] == s) & (results_df["Model"] == m)]
            if sub.empty:
                continue
            row = sub.iloc[0]
            roc = row["ROC-AUC"]
            pr = row["PR-AUC"]
            roc_ci = eval(row["ROC-AUC 95% CI"]) if isinstance(row["ROC-AUC 95% CI"], str) else row["ROC-AUC 95% CI"]
            pr_ci = eval(row["PR-AUC 95% CI"]) if isinstance(row["PR-AUC 95% CI"], str) else row["PR-AUC 95% CI"]

            roc_vals.append(roc)
            roc_errs_lo.append(max(0, roc - roc_ci[0]))
            roc_errs_hi.append(max(0, roc_ci[1] - roc))

            pr_vals.append(pr)
            pr_errs_lo.append(max(0, pr - pr_ci[0]))
            pr_errs_hi.append(max(0, pr_ci[1] - pr))

        pos = indices + (i - n_models / 2 + 0.5) * width
        yerr_roc = [roc_errs_lo, roc_errs_hi]
        yerr_pr = [pr_errs_lo, pr_errs_hi]

        ax_roc.bar(
            pos,
            roc_vals,
            width,
            yerr=yerr_roc,
            capsize=3,
            label=m,
            color=col,
            edgecolor="#2D3748",
            alpha=0.9,
        )
        ax_pr.bar(
            pos,
            pr_vals,
            width,
            yerr=yerr_pr,
            capsize=3,
            label=m,
            color=col,
            edgecolor="#2D3748",
            alpha=0.9,
        )

    # Configure ROC-AUC panel
    ax_roc.set_facecolor("#FFFFFF")
    ax_roc.set_xticks(indices)
    ax_roc.set_xticklabels(split_labels, fontsize=11, fontweight="600", color="#2D3748")
    ax_roc.set_ylabel("Test ROC-AUC", fontsize=12, fontweight="600", color="#2D3748")
    ax_roc.set_title("ROC-AUC Across Generalization Regimes", fontsize=13, fontweight="700", color="#1A202C", pad=12)
    ax_roc.set_ylim(0.40, 1.02)
    ax_roc.axhline(0.5, color="#E53E3E", linestyle=":", linewidth=1.5, alpha=0.7, label="Random (0.50)")
    ax_roc.grid(axis="y", linestyle="--", alpha=0.45, color="#CBD5E0")
    ax_roc.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0", loc="lower left", fontsize=8, ncol=2)

    # Configure PR-AUC panel
    ax_pr.set_facecolor("#FFFFFF")
    ax_pr.set_xticks(indices)
    ax_pr.set_xticklabels(split_labels, fontsize=11, fontweight="600", color="#2D3748")
    ax_pr.set_ylabel("Test PR-AUC", fontsize=12, fontweight="600", color="#2D3748")
    ax_pr.set_title("PR-AUC Across Generalization Regimes", fontsize=13, fontweight="700", color="#1A202C", pad=12)
    ax_pr.set_ylim(0.40, 1.02)
    ax_pr.grid(axis="y", linestyle="--", alpha=0.45, color="#CBD5E0")
    ax_pr.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0", loc="lower left", fontsize=8, ncol=2)

    fig.suptitle("Drug-Target Interaction Model Performance Across Four Evaluation Regimes",
                 fontsize=14.5, fontweight="700", color="#1A202C", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved to '{output_path}'")


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def train_and_evaluate(
    dataset_path: str = "bindingdb_human_dti_binary.tsv",
    splits_dir: str = "splits",
    features_dir: str = "features",
    results_dir: str = "results",
    train_size: int = 40000,
    seed: int = 42,
):
    start_total = time.time()
    preds_dir = os.path.join(results_dir, "preds")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    print("=" * 80)
    print("DTI Model Training and Four-Regime Evaluation")
    print("=" * 80)

    # 1. RAM Check & Subsampling Determination
    mem = psutil.virtual_memory()
    avail_ram_gb = mem.available / (1024**3)
    total_ram_gb = mem.total / (1024**3)
    print(f"System RAM: {total_ram_gb:.2f} GB total, {avail_ram_gb:.2f} GB available ({mem.percent}% used).")
    print(f"[*] Training sample size limit: {train_size:,} pairs per split (stratified, seed={seed}) for uniform comparison.\n")

    # 2. Load Dataset and Precomputed Features
    print("Loading dataset and precomputed features...")
    df = pd.read_csv(dataset_path, sep="\t", low_memory=False)
    drug_features, protein_features, drug_to_idx, prot_to_idx = load_features(features_dir)
    print(f"Dataset: {len(df):,} pairs loaded.")
    print(f"Drug features: {drug_features.shape}, Protein features: {protein_features.shape}\n")

    splits = ["random", "unseen_drugs", "unseen_proteins", "unseen_both"]
    all_results = []

    for s_idx, split_name in enumerate(splits, 1):
        print("=" * 80)
        print(f"[{s_idx}/4] Evaluating Regime: '{split_name.upper()}'")
        print("=" * 80)

        train_idx, val_idx, test_idx = load_split(split_name, splits_dir)
        y_train_full = df["label"].iloc[train_idx].values
        y_val = df["label"].iloc[val_idx].values
        y_test = df["label"].iloc[test_idx].values

        # Subsample training set to train_size for every split if larger
        if train_size > 0 and len(train_idx) > train_size:
            sub_tr_idx, _ = train_test_split(
                train_idx,
                train_size=train_size,
                stratify=y_train_full,
                random_state=seed,
            )
            train_idx_effective = np.sort(sub_tr_idx)
            y_train = df["label"].iloc[train_idx_effective].values
            print(f"Subsampled train set: {len(train_idx):,} -> {len(train_idx_effective):,} (val={len(val_idx):,}, test={len(test_idx):,})")
        else:
            train_idx_effective = train_idx
            y_train = y_train_full
            print(f"Train pairs: {len(train_idx_effective):,}, Val pairs: {len(val_idx):,}, Test pairs: {len(test_idx):,}")

        # Assemble features
        print(f"Building feature matrices for '{split_name}'...")
        t0_feat = time.time()
        X_train = build_pair_features(train_idx_effective, df, drug_features, protein_features, drug_to_idx, prot_to_idx)
        X_val = build_pair_features(val_idx, df, drug_features, protein_features, drug_to_idx, prot_to_idx)
        X_test = build_pair_features(test_idx, df, drug_features, protein_features, drug_to_idx, prot_to_idx)
        print(f"Feature assembly done in {time.time() - t0_feat:.2f}s (X_train: {X_train.shape}, {X_train.nbytes / (1024**2):.1f} MB)")

        split_predictions = {"test_idx": test_idx, "y_true": y_test}

        # -------------------------------------------------------------
        # (a) Dummy Baseline
        # -------------------------------------------------------------
        t0 = time.time()
        dummy_preds = run_dummy_baseline(y_train, len(y_test))
        t_dummy = time.time() - t0
        roc_d, ci_roc_d, pr_d, ci_pr_d = compute_metrics_and_bootstrap_ci(y_test, dummy_preds, seed=seed)
        split_predictions["dummy"] = dummy_preds
        all_results.append({
            "Split": split_name, "Model": "Dummy",
            "ROC-AUC": roc_d, "ROC-AUC 95% CI": ci_roc_d,
            "PR-AUC": pr_d, "PR-AUC 95% CI": ci_pr_d,
            "Train Time (s)": round(t_dummy, 3)
        })
        print(f"  [Dummy]           ROC-AUC: {roc_d:.4f} {ci_roc_d} | PR-AUC: {pr_d:.4f} {ci_pr_d} | Time: {t_dummy:.3f}s")

        # -------------------------------------------------------------
        # (b) Protein-Only Baseline
        # -------------------------------------------------------------
        t0 = time.time()
        prot_preds = run_protein_only_baseline(df, train_idx_effective, test_idx)
        t_prot = time.time() - t0
        roc_p, ci_roc_p, pr_p, ci_pr_p = compute_metrics_and_bootstrap_ci(y_test, prot_preds, seed=seed)
        split_predictions["protein_only"] = prot_preds
        all_results.append({
            "Split": split_name, "Model": "Protein-Only",
            "ROC-AUC": roc_p, "ROC-AUC 95% CI": ci_roc_p,
            "PR-AUC": pr_p, "PR-AUC 95% CI": ci_pr_p,
            "Train Time (s)": round(t_prot, 3)
        })
        print(f"  [Protein-Only]    ROC-AUC: {roc_p:.4f} {ci_roc_p} | PR-AUC: {pr_p:.4f} {ci_pr_p} | Time: {t_prot:.3f}s")

        # -------------------------------------------------------------
        # (c) Drug-Only Baseline
        # -------------------------------------------------------------
        t0 = time.time()
        drug_preds = run_drug_only_baseline(df, train_idx_effective, test_idx)
        t_drug = time.time() - t0
        roc_dr, ci_roc_dr, pr_dr, ci_pr_dr = compute_metrics_and_bootstrap_ci(y_test, drug_preds, seed=seed)
        split_predictions["drug_only"] = drug_preds
        all_results.append({
            "Split": split_name, "Model": "Drug-Only",
            "ROC-AUC": roc_dr, "ROC-AUC 95% CI": ci_roc_dr,
            "PR-AUC": pr_dr, "PR-AUC 95% CI": ci_pr_dr,
            "Train Time (s)": round(t_drug, 3)
        })
        print(f"  [Drug-Only]       ROC-AUC: {roc_dr:.4f} {ci_roc_dr} | PR-AUC: {pr_dr:.4f} {ci_pr_dr} | Time: {t_drug:.3f}s")

        # -------------------------------------------------------------
        # (d) Nearest-Protein Baseline
        # -------------------------------------------------------------
        t0 = time.time()
        nearest_prot_preds = run_nearest_protein_baseline(df, protein_features, prot_to_idx, train_idx_effective, test_idx)
        t_np = time.time() - t0
        roc_np, ci_roc_np, pr_np, ci_pr_np = compute_metrics_and_bootstrap_ci(y_test, nearest_prot_preds, seed=seed)
        split_predictions["nearest_protein"] = nearest_prot_preds
        all_results.append({
            "Split": split_name, "Model": "Nearest-Protein",
            "ROC-AUC": roc_np, "ROC-AUC 95% CI": ci_roc_np,
            "PR-AUC": pr_np, "PR-AUC 95% CI": ci_pr_np,
            "Train Time (s)": round(t_np, 3)
        })
        print(f"  [Nearest-Protein] ROC-AUC: {roc_np:.4f} {ci_roc_np} | PR-AUC: {pr_np:.4f} {ci_pr_np} | Time: {t_np:.3f}s")

        # -------------------------------------------------------------
        # (e) RF drug-features-only (Ablation: 2,048 ECFP4 fingerprint bits only)
        # -------------------------------------------------------------
        print(f"  Training RF drug-features-only (2048 fingerprint bits)...")
        rf_drug_preds, _, t_rf_drug = run_random_forest(X_train[:, :2048], y_train, X_test[:, :2048], seed=seed)
        roc_rfd, ci_roc_rfd, pr_rfd, ci_pr_rfd = compute_metrics_and_bootstrap_ci(y_test, rf_drug_preds, seed=seed)
        split_predictions["rf_drug_only"] = rf_drug_preds
        all_results.append({
            "Split": split_name, "Model": "RF drug-features-only",
            "ROC-AUC": roc_rfd, "ROC-AUC 95% CI": ci_roc_rfd,
            "PR-AUC": pr_rfd, "PR-AUC 95% CI": ci_pr_rfd,
            "Train Time (s)": round(t_rf_drug, 3)
        })
        print(f"  [RF drug-only]    ROC-AUC: {roc_rfd:.4f} {ci_roc_rfd} | PR-AUC: {pr_rfd:.4f} {ci_pr_rfd} | Time: {t_rf_drug:.2f}s")

        # -------------------------------------------------------------
        # (f) RF protein-features-only (Ablation: 420 composition features only)
        # -------------------------------------------------------------
        print(f"  Training RF protein-features-only (420 composition features)...")
        rf_prot_preds, _, t_rf_prot = run_random_forest(X_train[:, 2048:], y_train, X_test[:, 2048:], seed=seed)
        roc_rfp, ci_roc_rfp, pr_rfp, ci_pr_rfp = compute_metrics_and_bootstrap_ci(y_test, rf_prot_preds, seed=seed)
        split_predictions["rf_protein_only"] = rf_prot_preds
        all_results.append({
            "Split": split_name, "Model": "RF protein-features-only",
            "ROC-AUC": roc_rfp, "ROC-AUC 95% CI": ci_roc_rfp,
            "PR-AUC": pr_rfp, "PR-AUC 95% CI": ci_pr_rfp,
            "Train Time (s)": round(t_rf_prot, 3)
        })
        print(f"  [RF protein-only] ROC-AUC: {roc_rfp:.4f} {ci_roc_rfp} | PR-AUC: {pr_rfp:.4f} {ci_pr_rfp} | Time: {t_rf_prot:.2f}s")

        # -------------------------------------------------------------
        # (g) Random Forest (concatenated 2,468 features)
        # -------------------------------------------------------------
        print(f"  Training Random Forest (200 trees, balanced, n_jobs=-1)...")
        rf_preds, rf_model, t_rf = run_random_forest(X_train, y_train, X_test, seed=seed)
        roc_rf, ci_roc_rf, pr_rf, ci_pr_rf = compute_metrics_and_bootstrap_ci(y_test, rf_preds, seed=seed)
        split_predictions["random_forest"] = rf_preds
        all_results.append({
            "Split": split_name, "Model": "Random Forest",
            "ROC-AUC": roc_rf, "ROC-AUC 95% CI": ci_roc_rf,
            "PR-AUC": pr_rf, "PR-AUC 95% CI": ci_pr_rf,
            "Train Time (s)": round(t_rf, 3)
        })
        print(f"  [Random Forest]   ROC-AUC: {roc_rf:.4f} {ci_roc_rf} | PR-AUC: {pr_rf:.4f} {ci_pr_rf} | Time: {t_rf:.2f}s")

        # Save ONLY split-1 Random Forest model (compressed)
        if split_name == "random":
            rf_save_path = os.path.join(results_dir, "rf_split1.joblib")
            joblib.dump(rf_model, rf_save_path, compress=3)
            rf_size_mb = os.path.getsize(rf_save_path) / (1024**2)
            print(f"  [Model Saved] Compressed Split-1 RF model saved to '{rf_save_path}' ({rf_size_mb:.2f} MB).")

        # -------------------------------------------------------------
        # (h) PyTorch MLP
        # -------------------------------------------------------------
        print(f"  Training PyTorch MLP (2468->512->128->1, Adam lr=1e-3, early stopping)...")
        mlp_preds, t_mlp = run_pytorch_mlp(X_train, y_train, X_val, y_val, X_test, seed=seed)
        roc_mlp, ci_roc_mlp, pr_mlp, ci_pr_mlp = compute_metrics_and_bootstrap_ci(y_test, mlp_preds, seed=seed)
        split_predictions["mlp"] = mlp_preds
        all_results.append({
            "Split": split_name, "Model": "PyTorch MLP",
            "ROC-AUC": roc_mlp, "ROC-AUC 95% CI": ci_roc_mlp,
            "PR-AUC": pr_mlp, "PR-AUC 95% CI": ci_pr_mlp,
            "Train Time (s)": round(t_mlp, 3)
        })
        print(f"  [PyTorch MLP]     ROC-AUC: {roc_mlp:.4f} {ci_roc_mlp} | PR-AUC: {pr_mlp:.4f} {ci_pr_mlp} | Time: {t_mlp:.2f}s")

        # Save test predictions
        npz_pred_path = os.path.join(preds_dir, f"{split_name}_preds.npz")
        np.savez_compressed(
            npz_pred_path,
            test_idx=test_idx,
            y_true=y_test,
            dummy=dummy_preds,
            protein_only=prot_preds,
            drug_only=drug_preds,
            nearest_protein=nearest_prot_preds,
            rf_drug_only=rf_drug_preds,
            rf_protein_only=rf_prot_preds,
            random_forest=rf_preds,
            mlp=mlp_preds,
        )

        csv_pred_path = os.path.join(preds_dir, f"{split_name}_predictions.csv")
        pd.DataFrame(split_predictions).to_csv(csv_pred_path, index=False)
        print(f"  [Predictions] Saved test predictions to '{preds_dir}/' ({split_name}_preds.npz & .csv)\n")

    # 3. Save Summary Table
    table_path = os.path.join(results_dir, "four_regime_table.csv")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(table_path, index=False)
    print("=" * 80)
    print(f"Results table saved to '{table_path}'.")

    # 4. Generate Grouped Bar Chart
    chart_path = os.path.join(results_dir, "four_regime_chart.png")
    plot_four_regime_chart(results_df, chart_path)

    total_time = time.time() - start_total
    print("\n" + "=" * 95)
    print("FOUR-REGIME EVALUATION SUMMARY TABLE")
    print("=" * 95)

    # Format table for console display
    disp_df = results_df.copy()
    disp_df["ROC-AUC (95% CI)"] = [
        f"{r:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]" for r, ci in zip(disp_df["ROC-AUC"], disp_df["ROC-AUC 95% CI"])
    ]
    disp_df["PR-AUC (95% CI)"] = [
        f"{p:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]" for p, ci in zip(disp_df["PR-AUC"], disp_df["PR-AUC 95% CI"])
    ]
    disp_cols = ["Split", "Model", "ROC-AUC (95% CI)", "PR-AUC (95% CI)", "Train Time (s)"]
    print(disp_df[disp_cols].to_string(index=False))
    print("=" * 95)
    print(f"Total Execution Time: {total_time:.2f} seconds ({total_time / 60:.1f} minutes)")
    print("=" * 95)


def evaluate_split4_rf_seeds(
    dataset_path: str = "bindingdb_human_dti_binary.tsv",
    splits_dir: str = "splits",
    features_dir: str = "features",
    results_dir: str = "results",
    train_size: int = 150000,
    seeds: list[int] = [42, 1, 2],
):
    """
    Run full Random Forest on Split 4 (unseen drugs and proteins) across multiple seeds.
    Each seed performs its own 150k stratified subsample of the training set and seeds the RF.
    Reports individual ROC-AUC, PR-AUC, bootstrap 95% CIs, mean, and range.
    """
    print("=" * 80)
    print(f"SPLIT 4 RANDOM FOREST MULTI-SEED ROBUSTNESS EVALUATION (train_size={train_size:,})")
    print(f"Seeds: {seeds}")
    print("=" * 80)

    os.makedirs(results_dir, exist_ok=True)
    df = pd.read_csv(dataset_path, sep="\t", low_memory=False)
    drug_features, protein_features, drug_to_idx, prot_to_idx = load_features(features_dir)

    train_idx, val_idx, test_idx = load_split("unseen_both", splits_dir)
    y_train_full = df["label"].iloc[train_idx].values
    y_test = df["label"].iloc[test_idx].values

    print(f"Split 4 full train pairs: {len(train_idx):,}, Test pairs: {len(test_idx):,}")
    print(f"Building Split 4 test features (N={len(test_idx):,})...")
    X_test = build_pair_features(test_idx, df, drug_features, protein_features, drug_to_idx, prot_to_idx)

    seed_results = []

    for s in seeds:
        print(f"\n--- Running Seed {s} ---")
        t0 = time.time()
        if train_size > 0 and len(train_idx) > train_size:
            sub_tr_idx, _ = train_test_split(
                train_idx,
                train_size=train_size,
                stratify=y_train_full,
                random_state=s,
            )
            sub_tr_idx = np.sort(sub_tr_idx)
        else:
            sub_tr_idx = train_idx

        y_train = df["label"].iloc[sub_tr_idx].values
        print(f"  Subsampled training pairs: {len(sub_tr_idx):,} (pos rate: {np.mean(y_train)*100:.2f}%)")

        t_feat = time.time()
        X_train = build_pair_features(sub_tr_idx, df, drug_features, protein_features, drug_to_idx, prot_to_idx)
        print(f"  X_train built in {time.time() - t_feat:.2f}s")

        t_rf = time.time()
        rf_preds, rf_model, fit_time = run_random_forest(X_train, y_train, X_test, seed=s)
        print(f"  RF fitted in {fit_time:.2f}s")

        roc_pt, roc_ci, pr_pt, pr_ci = compute_metrics_and_bootstrap_ci(y_test, rf_preds, seed=s)
        total_seed_time = time.time() - t0

        print(f"  Seed {s} Results: ROC-AUC = {roc_pt:.4f} {roc_ci} | PR-AUC = {pr_pt:.4f} {pr_ci} (Time: {total_seed_time:.1f}s)")

        seed_results.append({
            "Seed": s,
            "Train Pairs": len(sub_tr_idx),
            "ROC-AUC": roc_pt,
            "ROC-AUC 95% CI": roc_ci,
            "PR-AUC": pr_pt,
            "PR-AUC 95% CI": pr_ci,
            "Fit Time (s)": round(fit_time, 2),
        })

    res_df = pd.DataFrame(seed_results)
    rocs = res_df["ROC-AUC"].values
    prs = res_df["PR-AUC"].values

    mean_roc = float(np.mean(rocs))
    min_roc = float(np.min(rocs))
    max_roc = float(np.max(rocs))
    range_roc = max_roc - min_roc
    std_roc = float(np.std(rocs, ddof=1)) if len(rocs) > 1 else 0.0

    mean_pr = float(np.mean(prs))
    min_pr = float(np.min(prs))
    max_pr = float(np.max(prs))
    range_pr = max_pr - min_pr

    print("\n" + "=" * 80)
    print("SUMMARY OF SPLIT 4 RANDOM FOREST MULTI-SEED RUNS")
    print("=" * 80)
    for r in seed_results:
        print(f"Seed {r['Seed']:2d}: ROC-AUC = {r['ROC-AUC']:.4f} [{r['ROC-AUC 95% CI'][0]:.4f}, {r['ROC-AUC 95% CI'][1]:.4f}] | PR-AUC = {r['PR-AUC']:.4f} | Fit Time = {r['Fit Time (s)']}s")
    print("-" * 80)
    print(f"ROC-AUC  Mean: {mean_roc:.4f} | Range: [{min_roc:.4f}, {max_roc:.4f}] (span = {range_roc:.4f}, std = {std_roc:.4f})")
    print(f"PR-AUC   Mean: {mean_pr:.4f} | Range: [{min_pr:.4f}, {max_pr:.4f}] (span = {range_pr:.4f})")
    print("=" * 80)

    # Save summary to results
    out_path = os.path.join(results_dir, "split4_rf_seed_robustness.csv")
    res_df.to_csv(out_path, index=False)
    print(f"Saved robustness table to '{out_path}'")
    return res_df, (mean_roc, min_roc, max_roc, range_roc)


# -----------------------------------------------------------------------------
# Web Export Mode
# -----------------------------------------------------------------------------
def export_web_mode(
    dataset_path: str = "bindingdb_human_dti_binary.tsv",
    splits_dir: str = "splits",
    features_dir: str = "features",
    results_dir: str = "results",
    web_dir: str = "web",
    train_size: int = 150000,
    seed: int = 42,
):
    """
    Train PyTorch MLP on Split 1 with 150,000 pairs and export web assets:
    (1) model.onnx - trained MLP exported with torch.onnx.export, input shape (1, 2468),
        opset 17, dynamic batch axis, after model.eval(). Verified with onnxruntime (<1e-5 diff).
    (2) targets.json - top 200 proteins with most pairs, target_name, uniprot_id, pair count,
        training positive rate, 420 composition features rounded to 5 decimals.
    (3) domain.json - 30,000-compound sample of split-1 training set, ECFP4 fingerprints bit-packed
        with np.packbits and base64-encoded.
    (4) meta.json - four-regime ROC-AUC table, similarity bins, and seed-robustness figures.
    """
    os.makedirs(web_dir, exist_ok=True)

    print("=" * 80)
    print("DTI Web Export: Training MLP on Split 1 and Generating Web Assets")
    print(f"Target directory: '{web_dir}' | Training size: {train_size:,} | Seed: {seed}")
    print("=" * 80)

    # 1. Load Dataset and Precomputed Features
    print("Loading dataset and precomputed features...")
    df = pd.read_csv(dataset_path, sep="\t", low_memory=False)
    drug_features, protein_features, drug_to_idx, prot_to_idx = load_features(features_dir)
    print(f"Loaded {len(df):,} pairs. Drug features: {drug_features.shape}, Protein features: {protein_features.shape}")

    # 2. Load Split 1 (Random Pairs)
    train_idx, val_idx, test_idx = load_split("random", splits_dir)
    y_train_full = df["label"].iloc[train_idx].values
    y_val = df["label"].iloc[val_idx].values
    y_test = df["label"].iloc[test_idx].values

    if train_size > 0 and len(train_idx) > train_size:
        sub_tr_idx, _ = train_test_split(
            train_idx,
            train_size=train_size,
            stratify=y_train_full,
            random_state=seed,
        )
        train_idx_effective = np.sort(sub_tr_idx)
        y_train = df["label"].iloc[train_idx_effective].values
        print(f"Subsampled Split 1 training set: {len(train_idx):,} -> {len(train_idx_effective):,} pairs (val: {len(val_idx):,}, test: {len(test_idx):,})")
    else:
        train_idx_effective = train_idx
        y_train = y_train_full
        print(f"Split 1 training set: {len(train_idx_effective):,} pairs")

    # 3. Assemble Features
    print("Assembling feature matrices for Split 1...")
    X_train = build_pair_features(train_idx_effective, df, drug_features, protein_features, drug_to_idx, prot_to_idx)
    X_val = build_pair_features(val_idx, df, drug_features, protein_features, drug_to_idx, prot_to_idx)
    X_test = build_pair_features(test_idx, df, drug_features, protein_features, drug_to_idx, prot_to_idx)
    print(f"Features assembled: X_train {X_train.shape}, X_val {X_val.shape}, X_test {X_test.shape}")

    # 4. Train PyTorch MLP
    print("\nTraining PyTorch MLP (2468->512->128->1, Adam lr=1e-3, batch=512, early stopping)...")
    mlp_preds, t_mlp, model = run_pytorch_mlp(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        seed=seed,
        return_model=True,
    )
    roc_mlp, ci_roc_mlp, pr_mlp, ci_pr_mlp = compute_metrics_and_bootstrap_ci(y_test, mlp_preds, seed=seed)
    print(f"MLP Training Complete in {t_mlp:.2f}s | Test ROC-AUC: {roc_mlp:.4f} | PR-AUC: {pr_mlp:.4f}")

    # 5. Export (1): model.onnx
    onnx_path = os.path.join(web_dir, "model.onnx")
    print(f"\n[1/4] Exporting ONNX model to '{onnx_path}'...")
    model.eval()
    dummy_input = torch.zeros(1, 2468, dtype=torch.float32)

    export_kwargs = {
        "opset_version": 17,
        "input_names": ["input"],
        "output_names": ["output"],
        "dynamic_axes": {"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    }
    import inspect
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        **export_kwargs,
    )
    print("Exported model.onnx with opset 17 and dynamic batch axis.")

    # Verify ONNX export against PyTorch on 500 test pairs
    print("Verifying ONNX export against PyTorch model on 500 test pairs...")
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path)

    test_500 = X_test[:500]
    model.eval()
    with torch.no_grad():
        torch_out = model(torch.from_numpy(test_500)).cpu().numpy().squeeze()

    input_name = session.get_inputs()[0].name
    ort_out = session.run(None, {input_name: test_500})[0].squeeze()

    max_abs_diff = float(np.max(np.abs(torch_out - ort_out)))
    print(f"Maximum absolute difference between PyTorch and ONNX Runtime: {max_abs_diff:.8e}")
    if max_abs_diff >= 1e-5:
        raise ValueError(f"ONNX verification failed: difference {max_abs_diff:.8e} >= 1e-5")
    print("ONNX verification PASSED: difference is below 1e-5.")

    # 6. Export (2): targets.json
    targets_path = os.path.join(web_dir, "targets.json")
    print(f"\n[2/4] Generating '{targets_path}' (200 proteins with most pairs)...")
    top_200_seqs = df["protein_sequence"].value_counts().head(200).index.tolist()

    train_df = df.iloc[train_idx_effective]
    train_pos_rates = train_df.groupby("protein_sequence")["label"].mean().to_dict()
    global_train_pos = float(train_df["label"].mean())
    pair_counts = df["protein_sequence"].value_counts().to_dict()

    targets_data = []
    for seq in top_200_seqs:
        sub = df[df["protein_sequence"] == seq].iloc[0]
        t_name = str(sub["target_name"]) if pd.notna(sub["target_name"]) else "Unnamed target"
        u_id = str(sub["uniprot_id"]) if pd.notna(sub["uniprot_id"]) else ""
        cnt = int(pair_counts.get(seq, 0))
        tr_pos = round(float(train_pos_rates.get(seq, global_train_pos)), 5)

        p_idx = prot_to_idx[seq]
        comp_feats = [round(float(v), 5) for v in protein_features[p_idx]]

        targets_data.append({
            "target_name": t_name,
            "uniprot_id": u_id,
            "pair_count": cnt,
            "pairs": cnt,
            "training_positive_rate": tr_pos,
            "pos_rate": tr_pos,
            "composition_features": comp_feats,
            "features": comp_feats,
            "protein_sequence": seq,
        })

    with open(targets_path, "w", encoding="utf-8") as f:
        json.dump(targets_data, f, indent=2)
    print(f"Saved {len(targets_data)} targets to '{targets_path}'.")

    # 7. Export (3): domain.json
    domain_path = os.path.join(web_dir, "domain.json")
    print(f"\n[3/4] Generating '{domain_path}' (30,000-compound sample from Split 1 train)...")
    train_smiles = np.unique(train_df["canonical_smiles"].values)
    rng = np.random.RandomState(seed)
    if len(train_smiles) > 30000:
        chosen_indices = rng.choice(len(train_smiles), size=30000, replace=False)
        sample_smiles = train_smiles[chosen_indices]
    else:
        sample_smiles = train_smiles

    sample_rows = [drug_to_idx[s] for s in sample_smiles]
    sample_fps = drug_features[sample_rows]  # (30000, 2048) uint8
    packed_fps = np.packbits(sample_fps, axis=1)  # (30000, 256) uint8

    domain_list = [base64.b64encode(packed_fps[i]).decode("ascii") for i in range(len(packed_fps))]

    with open(domain_path, "w", encoding="utf-8") as f:
        json.dump(domain_list, f)
    print(f"Saved {len(domain_list):,} bit-packed base64 fingerprints to '{domain_path}'.")

    # 8. Export (4): meta.json
    meta_path = os.path.join(web_dir, "meta.json")
    print(f"\n[4/4] Generating '{meta_path}' (four-regime table, similarity bins, seed robustness)...")
    four_regime_csv = os.path.join(results_dir, "four_regime_table.csv")
    if os.path.exists(four_regime_csv):
        fr_df = pd.read_csv(four_regime_csv)
        fr_records = fr_df.to_dict(orient="records")
    else:
        fr_records = []

    from analyse import run_similarity_analysis
    try:
        binned_results, _ = run_similarity_analysis(
            df=df,
            prot_feats=protein_features,
            prot_to_idx=prot_to_idx,
            drug_feats=drug_features,
            drug_to_idx=drug_to_idx,
            results_dir=results_dir,
            train_size=train_size,
            seed=seed,
        )
        sim_bins_dict = {
            k: v.to_dict(orient="records") for k, v in binned_results.items()
        }
    except Exception as e:
        print(f"Warning: could not recompute similarity bins dynamically ({e}); proceeding without dynamic bins...")
        sim_bins_dict = {}

    seed_csv = os.path.join(results_dir, "split4_rf_seed_robustness.csv")
    if os.path.exists(seed_csv):
        sr_df = pd.read_csv(seed_csv)
        sr_records = sr_df.to_dict(orient="records")
    else:
        sr_records = []

    meta_obj = {
        "four_regime_table": fr_records,
        "regimes": fr_records,
        "similarity_bins": sim_bins_dict,
        "similarity": sim_bins_dict,
        "seed_robustness": sr_records,
        "split4_rf_seed_robustness": sr_records,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_obj, f, indent=2)
    print(f"Saved metadata to '{meta_path}'.")

    # 9. Print file sizes
    print("\n" + "=" * 80)
    print("WEB ASSETS GENERATED IN 'web/' FOLDER")
    print("=" * 80)
    files_to_report = [
        "model.onnx",
        "targets.json",
        "domain.json",
        "meta.json",
    ]
    for fn in files_to_report:
        fp = os.path.join(web_dir, fn)
        if os.path.exists(fp):
            sz = os.path.getsize(fp)
            print(f"  {fn:15s}: {sz:>10,} bytes ({sz / (1024**2):>6.2f} MB)")
        else:
            print(f"  {fn:15s}: NOT FOUND")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate DTI benchmark models.")
    parser.add_argument("--dataset", default="bindingdb_human_dti_binary.tsv", help="Dataset TSV path")
    parser.add_argument("--splits", default="splits", help="Splits directory")
    parser.add_argument("--features", default="features", help="Features directory")
    parser.add_argument("--results", default="results", help="Results directory")
    parser.add_argument("--web", default="web", help="Web export directory")
    parser.add_argument("--train-size", "--subsample", dest="train_size", type=int, default=40000, help="Uniform training sample size per split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-split4-seeds", action="store_true", help="Run multi-seed evaluation of RF on Split 4")
    parser.add_argument("--split4-seeds", type=int, nargs="+", default=[42, 1, 2], help="Seeds to evaluate on Split 4")
    parser.add_argument("--export-web", action="store_true", help="Train MLP on split 1 with 150k pairs and export web/ assets")
    args = parser.parse_args()

    if args.export_web:
        train_sz = args.train_size if args.train_size != 40000 else 150000
        export_web_mode(
            dataset_path=args.dataset,
            splits_dir=args.splits,
            features_dir=args.features,
            results_dir=args.results,
            web_dir=args.web,
            train_size=train_sz,
            seed=args.seed,
        )
    elif args.eval_split4_seeds:
        evaluate_split4_rf_seeds(
            dataset_path=args.dataset,
            splits_dir=args.splits,
            features_dir=args.features,
            results_dir=args.results,
            train_size=args.train_size,
            seeds=args.split4_seeds,
        )
    else:
        train_and_evaluate(
            dataset_path=args.dataset,
            splits_dir=args.splits,
            features_dir=args.features,
            results_dir=args.results,
            train_size=args.train_size,
            seed=args.seed,
        )
