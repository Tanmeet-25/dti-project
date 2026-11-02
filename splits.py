#!/usr/bin/env python3
"""
Create reproducible train/validation/test splits (80/20 train/test, 10% val carved from train)
for bindingdb_human_dti_binary.tsv with a fixed seed of 42.

Splits created:
1. Random pairs
2. Unseen drugs (GroupShuffleSplit on canonical_smiles)
3. Unseen proteins (GroupShuffleSplit on uniprot_id, falling back to sequence)
4. Unseen drugs AND proteins (hold out 20% drugs and 20% proteins; test = both held out,
   train = neither held out, discard mixed pairs; 10% val carved similarly)

Saves row indices only to the splits/ directory (.npy and .npz).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit


def get_protein_groups(df: pd.DataFrame) -> pd.Series:
    """Group by uniprot_id, falling back to sequence where uniprot_id is missing/empty."""
    return df["uniprot_id"].fillna(df["protein_sequence"]).astype(str)


def assert_split_integrity(
    split_name: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    labels: np.ndarray,
    drugs: np.ndarray,
    prot_groups: np.ndarray,
    check_drug_leakage: bool = False,
    check_protein_leakage: bool = False,
):
    """Enforce integrity assertions on the split indices."""
    # 1. Non-empty checks
    assert len(train_idx) > 0, f"[{split_name}] Train set is empty!"
    assert len(val_idx) > 0, f"[{split_name}] Validation set is empty!"
    assert len(test_idx) > 0, f"[{split_name}] Test set is empty!"

    # 2. Both labels present in test and val
    test_labels = np.unique(labels[test_idx])
    val_labels = np.unique(labels[val_idx])
    train_labels = np.unique(labels[train_idx])
    assert len(test_labels) == 2 and set(test_labels) == {0, 1}, f"[{split_name}] Test set lacks both labels: {test_labels}"
    assert len(val_labels) == 2 and set(val_labels) == {0, 1}, f"[{split_name}] Val set lacks both labels: {val_labels}"
    assert len(train_labels) == 2 and set(train_labels) == {0, 1}, f"[{split_name}] Train set lacks both labels: {train_labels}"

    # 3. Disjoint index sets
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)
    assert len(train_set.intersection(val_set)) == 0, f"[{split_name}] Train and Val indices overlap!"
    assert len(train_set.intersection(test_set)) == 0, f"[{split_name}] Train and Test indices overlap!"
    assert len(val_set.intersection(test_set)) == 0, f"[{split_name}] Val and Test indices overlap!"

    # 4. Entity leakage checks
    if check_drug_leakage:
        train_drugs = set(drugs[train_idx])
        test_drugs = set(drugs[test_idx])
        drug_overlap = train_drugs.intersection(test_drugs)
        assert len(drug_overlap) == 0, f"[{split_name}] Drug leakage detected between train and test ({len(drug_overlap)} overlapping drugs)!"

    if check_protein_leakage:
        train_prots = set(prot_groups[train_idx])
        test_prots = set(prot_groups[test_idx])
        prot_overlap = train_prots.intersection(test_prots)
        assert len(prot_overlap) == 0, f"[{split_name}] Protein leakage detected between train and test ({len(prot_overlap)} overlapping proteins)!"


def make_random_split(all_indices: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split 1: Random pairs (80/20 train/test, 10% of train for val)."""
    train_val_idx, test_idx = train_test_split(all_indices, test_size=0.20, random_state=seed)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.10, random_state=seed)
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def make_unseen_drugs_split(
    all_indices: np.ndarray, drugs: np.ndarray, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split 2: Unseen drugs using GroupShuffleSplit on canonical_smiles."""
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_val_sub, test_sub = next(gss_test.split(all_indices, groups=drugs))
    train_val_idx = all_indices[train_val_sub]
    test_idx = all_indices[test_sub]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
    train_sub, val_sub = next(gss_val.split(train_val_idx, groups=drugs[train_val_idx]))
    train_idx = train_val_idx[train_sub]
    val_idx = train_val_idx[val_sub]

    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def make_unseen_proteins_split(
    all_indices: np.ndarray, prot_groups: np.ndarray, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split 3: Unseen proteins using GroupShuffleSplit on protein groups."""
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_val_sub, test_sub = next(gss_test.split(all_indices, groups=prot_groups))
    train_val_idx = all_indices[train_val_sub]
    test_idx = all_indices[test_sub]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
    train_sub, val_sub = next(gss_val.split(train_val_idx, groups=prot_groups[train_val_idx]))
    train_idx = train_val_idx[train_sub]
    val_idx = train_val_idx[val_sub]

    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def make_unseen_both_split(
    all_indices: np.ndarray,
    drugs: np.ndarray,
    prot_groups: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Split 4: Unseen drugs AND proteins.
    Hold out 20% of unique drugs and 20% of unique proteins.
    Test: pairs where both drug and protein are held out.
    Train candidate: pairs where neither is held out.
    Mixed pairs: discarded.
    From train candidate, carve 10% val set using the same grouping logic.
    """
    unique_drugs = np.sort(np.unique(drugs))
    unique_prots = np.sort(np.unique(prot_groups))

    # Hold out 20% test drugs and 20% test proteins
    tr_drugs_pool, test_drugs = train_test_split(unique_drugs, test_size=0.20, random_state=seed)
    tr_prots_pool, test_prots = train_test_split(unique_prots, test_size=0.20, random_state=seed)

    set_test_drugs = set(test_drugs)
    set_test_prots = set(test_prots)
    set_tr_drugs_pool = set(tr_drugs_pool)
    set_tr_prots_pool = set(tr_prots_pool)

    # Fast boolean masks
    is_d_test = pd.Series(drugs).isin(set_test_drugs).values
    is_p_test = pd.Series(prot_groups).isin(set_test_prots).values
    is_d_tr_pool = pd.Series(drugs).isin(set_tr_drugs_pool).values
    is_p_tr_pool = pd.Series(prot_groups).isin(set_tr_prots_pool).values

    test_mask = is_d_test & is_p_test
    train_pool_mask = is_d_tr_pool & is_p_tr_pool

    # Mixed pairs discarded in test/train-pool split
    discard_test_pool = len(all_indices) - (test_mask.sum() + train_pool_mask.sum())

    # Carve 10% val from train pool using same grouping (10% val drugs, 10% val proteins)
    final_tr_drugs, val_drugs = train_test_split(tr_drugs_pool, test_size=0.10, random_state=seed)
    final_tr_prots, val_prots = train_test_split(tr_prots_pool, test_size=0.10, random_state=seed)

    set_val_drugs = set(val_drugs)
    set_val_prots = set(val_prots)
    set_final_tr_drugs = set(final_tr_drugs)
    set_final_tr_prots = set(final_tr_prots)

    is_d_val = pd.Series(drugs).isin(set_val_drugs).values
    is_p_val = pd.Series(prot_groups).isin(set_val_prots).values
    is_d_final_tr = pd.Series(drugs).isin(set_final_tr_drugs).values
    is_p_final_tr = pd.Series(prot_groups).isin(set_final_tr_prots).values

    val_mask = train_pool_mask & is_d_val & is_p_val
    final_train_mask = train_pool_mask & is_d_final_tr & is_p_final_tr

    discard_val_pool = train_pool_mask.sum() - (val_mask.sum() + final_train_mask.sum())
    total_discarded = discard_test_pool + discard_val_pool

    train_idx = all_indices[final_train_mask]
    val_idx = all_indices[val_mask]
    test_idx = all_indices[test_mask]

    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx), int(total_discarded)


def save_split_indices(
    out_dir: str,
    split_name: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
):
    """Save split row indices to disk as individual .npy files and compressed .npz bundle."""
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, f"{split_name}_train.npy"), train_idx)
    np.save(os.path.join(out_dir, f"{split_name}_val.npy"), val_idx)
    np.save(os.path.join(out_dir, f"{split_name}_test.npy"), test_idx)

    np.savez_compressed(
        os.path.join(out_dir, f"{split_name}.npz"),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train=train_idx,
        val=val_idx,
        test=test_idx,
    )


def load_split(split_name: str, splits_dir: str = "splits") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience helper to load row indices for a split."""
    npz_path = os.path.join(splits_dir, f"{split_name}.npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        return data["train_idx"], data["val_idx"], data["test_idx"]

    train_path = os.path.join(splits_dir, f"{split_name}_train.npy")
    val_path = os.path.join(splits_dir, f"{split_name}_val.npy")
    test_path = os.path.join(splits_dir, f"{split_name}_test.npy")
    return np.load(train_path), np.load(val_path), np.load(test_path)


def create_all_splits(
    dataset_path: str = "bindingdb_human_dti_binary.tsv",
    out_dir: str = "splits",
    seed: int = 42,
):
    print("=" * 80)
    print("Generating DTI Benchmark Train / Val / Test Splits (Seed = 42)")
    print("=" * 80)

    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading '{dataset_path}'...")
    df = pd.read_csv(dataset_path, sep="\t", low_memory=False)
    n_total = len(df)
    all_indices = np.arange(n_total)

    labels = df["label"].values
    drugs = df["canonical_smiles"].values
    prot_groups = get_protein_groups(df).values

    print(f"Total dataset pairs: {n_total:,}")
    print(f"Unique drugs:        {len(np.unique(drugs)):,}")
    print(f"Unique protein groups: {len(np.unique(prot_groups)):,}\n")

    summary_records = []

    # -------------------------------------------------------------------------
    # Split 1: Random pairs
    # -------------------------------------------------------------------------
    print("1. Creating Split 1: Random pairs (80/20)...")
    s1_tr, s1_va, s1_te = make_random_split(all_indices, seed=seed)
    assert_split_integrity("random", s1_tr, s1_va, s1_te, labels, drugs, prot_groups)
    save_split_indices(out_dir, "random", s1_tr, s1_va, s1_te)
    summary_records.append({
        "Split": "1. Random Pairs",
        "Train Pairs": len(s1_tr),
        "Train %1": f"{labels[s1_tr].mean() * 100:.2f}%",
        "Val Pairs": len(s1_va),
        "Val %1": f"{labels[s1_va].mean() * 100:.2f}%",
        "Test Pairs": len(s1_te),
        "Test %1": f"{labels[s1_te].mean() * 100:.2f}%",
        "Discarded Pairs": 0,
    })

    # -------------------------------------------------------------------------
    # Split 2: Unseen drugs
    # -------------------------------------------------------------------------
    print("2. Creating Split 2: Unseen drugs (GroupShuffleSplit on canonical_smiles)...")
    s2_tr, s2_va, s2_te = make_unseen_drugs_split(all_indices, drugs, seed=seed)
    assert_split_integrity("unseen_drugs", s2_tr, s2_va, s2_te, labels, drugs, prot_groups, check_drug_leakage=True)
    save_split_indices(out_dir, "unseen_drugs", s2_tr, s2_va, s2_te)
    summary_records.append({
        "Split": "2. Unseen Drugs",
        "Train Pairs": len(s2_tr),
        "Train %1": f"{labels[s2_tr].mean() * 100:.2f}%",
        "Val Pairs": len(s2_va),
        "Val %1": f"{labels[s2_va].mean() * 100:.2f}%",
        "Test Pairs": len(s2_te),
        "Test %1": f"{labels[s2_te].mean() * 100:.2f}%",
        "Discarded Pairs": 0,
    })

    # -------------------------------------------------------------------------
    # Split 3: Unseen proteins
    # -------------------------------------------------------------------------
    print("3. Creating Split 3: Unseen proteins (GroupShuffleSplit on uniprot_id)...")
    s3_tr, s3_va, s3_te = make_unseen_proteins_split(all_indices, prot_groups, seed=seed)
    assert_split_integrity("unseen_proteins", s3_tr, s3_va, s3_te, labels, drugs, prot_groups, check_protein_leakage=True)
    save_split_indices(out_dir, "unseen_proteins", s3_tr, s3_va, s3_te)
    summary_records.append({
        "Split": "3. Unseen Proteins",
        "Train Pairs": len(s3_tr),
        "Train %1": f"{labels[s3_tr].mean() * 100:.2f}%",
        "Val Pairs": len(s3_va),
        "Val %1": f"{labels[s3_va].mean() * 100:.2f}%",
        "Test Pairs": len(s3_te),
        "Test %1": f"{labels[s3_te].mean() * 100:.2f}%",
        "Discarded Pairs": 0,
    })

    # -------------------------------------------------------------------------
    # Split 4: Unseen drugs AND proteins
    # -------------------------------------------------------------------------
    print("4. Creating Split 4: Unseen drugs AND proteins (20% holdout each)...")
    s4_tr, s4_va, s4_te, s4_discarded = make_unseen_both_split(all_indices, drugs, prot_groups, seed=seed)
    assert_split_integrity(
        "unseen_both",
        s4_tr,
        s4_va,
        s4_te,
        labels,
        drugs,
        prot_groups,
        check_drug_leakage=True,
        check_protein_leakage=True,
    )
    save_split_indices(out_dir, "unseen_both", s4_tr, s4_va, s4_te)
    summary_records.append({
        "Split": "4. Unseen Both",
        "Train Pairs": len(s4_tr),
        "Train %1": f"{labels[s4_tr].mean() * 100:.2f}%",
        "Val Pairs": len(s4_va),
        "Val %1": f"{labels[s4_va].mean() * 100:.2f}%",
        "Test Pairs": len(s4_te),
        "Test %1": f"{labels[s4_te].mean() * 100:.2f}%",
        "Discarded Pairs": s4_discarded,
    })

    print("\n" + "=" * 90)
    print("BENCHMARK SPLIT SIZES AND LABEL DISTRIBUTIONS (80/20 Train/Test, 10% Val from Train)")
    print("=" * 90)

    summary_df = pd.DataFrame(summary_records)
    print(summary_df.to_string(index=False))
    print("=" * 90)
    print(f"Split 4 discarded pairs detail:")
    print(f"  - Total discarded mixed pairs: {s4_discarded:,} ({s4_discarded / n_total * 100:.2f}% of total dataset)")
    print(f"  - Clean partitioned pairs:     {len(s4_tr) + len(s4_va) + len(s4_te):,} ({100 - s4_discarded / n_total * 100:.2f}%)")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create DTI train/val/test splits.")
    parser.add_argument("--input", default="bindingdb_human_dti_binary.tsv", help="Input dataset path")
    parser.add_argument("--outdir", default="splits", help="Output directory for split index arrays")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    create_all_splits(dataset_path=args.input, out_dir=args.outdir, seed=args.seed)
