"""
Test suite for DTI dataset benchmark splits:
Validates integrity, disjointness, label diversity, and entity separation.
"""

import os
import pytest
import numpy as np
import pandas as pd
from splits import load_split, get_protein_groups


@pytest.fixture(scope="module")
def dataset():
    data_path = "bindingdb_human_dti_binary.tsv"
    assert os.path.exists(data_path), f"Dataset not found at {data_path}"
    df = pd.read_csv(data_path, sep="\t", low_memory=False)
    prot_groups = get_protein_groups(df).values
    return {
        "df": df,
        "labels": df["label"].values,
        "drugs": df["canonical_smiles"].values,
        "prot_groups": prot_groups,
        "sequences": df["protein_sequence"].values,
        "n_total": len(df),
    }


@pytest.mark.parametrize("split_name", ["random", "unseen_drugs", "unseen_proteins", "unseen_both"])
def test_split_files_exist(split_name):
    splits_dir = "splits"
    assert os.path.exists(os.path.join(splits_dir, f"{split_name}.npz")), f"{split_name}.npz missing"
    assert os.path.exists(os.path.join(splits_dir, f"{split_name}_train.npy")), f"{split_name}_train.npy missing"
    assert os.path.exists(os.path.join(splits_dir, f"{split_name}_val.npy")), f"{split_name}_val.npy missing"
    assert os.path.exists(os.path.join(splits_dir, f"{split_name}_test.npy")), f"{split_name}_test.npy missing"


@pytest.mark.parametrize("split_name", ["random", "unseen_drugs", "unseen_proteins", "unseen_both"])
def test_split_non_empty_and_valid_indices(split_name, dataset):
    train_idx, val_idx, test_idx = load_split(split_name)
    n_total = dataset["n_total"]

    # Non-empty
    assert len(train_idx) > 0, f"Train set empty for {split_name}"
    assert len(val_idx) > 0, f"Val set empty for {split_name}"
    assert len(test_idx) > 0, f"Test set empty for {split_name}"

    # Valid index bounds
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        assert np.all(idx >= 0) and np.all(idx < n_total), f"Indices out of bounds in {split_name} {name}"
        assert len(idx) == len(np.unique(idx)), f"Duplicate indices found in {split_name} {name}"

    # Mutually disjoint indices
    s_tr, s_va, s_te = set(train_idx), set(val_idx), set(test_idx)
    assert len(s_tr.intersection(s_va)) == 0, f"Overlap between train and val in {split_name}"
    assert len(s_tr.intersection(s_te)) == 0, f"Overlap between train and test in {split_name}"
    assert len(s_va.intersection(s_te)) == 0, f"Overlap between val and test in {split_name}"


@pytest.mark.parametrize("split_name", ["random", "unseen_drugs", "unseen_proteins", "unseen_both"])
def test_both_labels_present(split_name, dataset):
    train_idx, val_idx, test_idx = load_split(split_name)
    labels = dataset["labels"]

    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        subset_labels = np.unique(labels[idx])
        assert len(subset_labels) == 2, f"{split_name} {name} does not have 2 classes: {subset_labels}"
        assert set(subset_labels) == {0, 1}, f"{split_name} {name} classes are not {{0, 1}}: {subset_labels}"


def test_unseen_drugs_no_leakage(dataset):
    """Assert no drug appears in both train and test for unseen_drugs split."""
    train_idx, val_idx, test_idx = load_split("unseen_drugs")
    drugs = dataset["drugs"]

    train_drugs = set(drugs[train_idx])
    val_drugs = set(drugs[val_idx])
    test_drugs = set(drugs[test_idx])

    assert len(train_drugs.intersection(test_drugs)) == 0, "Drug leakage between train and test in unseen_drugs!"
    assert len(val_drugs.intersection(test_drugs)) == 0, "Drug leakage between val and test in unseen_drugs!"
    assert len(train_drugs.intersection(val_drugs)) == 0, "Drug leakage between train and val in unseen_drugs!"


def test_unseen_proteins_no_leakage(dataset):
    """Assert no protein appears in both train and test for unseen_proteins split."""
    train_idx, val_idx, test_idx = load_split("unseen_proteins")
    prot_groups = dataset["prot_groups"]
    sequences = dataset["sequences"]

    train_prots = set(prot_groups[train_idx])
    val_prots = set(prot_groups[val_idx])
    test_prots = set(prot_groups[test_idx])

    assert len(train_prots.intersection(test_prots)) == 0, "Protein leakage between train and test in unseen_proteins!"
    assert len(val_prots.intersection(test_prots)) == 0, "Protein leakage between val and test in unseen_proteins!"
    assert len(train_prots.intersection(val_prots)) == 0, "Protein leakage between train and val in unseen_proteins!"

    # Also assert sequence-level leakage is zero
    assert len(set(sequences[train_idx]).intersection(set(sequences[test_idx]))) == 0, "Sequence leakage between train and test!"


def test_unseen_both_no_leakage(dataset):
    """Assert no drug AND no protein appears in both train and test for unseen_both split."""
    train_idx, val_idx, test_idx = load_split("unseen_both")
    drugs = dataset["drugs"]
    prot_groups = dataset["prot_groups"]
    sequences = dataset["sequences"]

    # Drug leakage check
    train_drugs = set(drugs[train_idx])
    test_drugs = set(drugs[test_idx])
    assert len(train_drugs.intersection(test_drugs)) == 0, "Drug leakage between train and test in unseen_both!"

    # Protein leakage check
    train_prots = set(prot_groups[train_idx])
    test_prots = set(prot_groups[test_idx])
    assert len(train_prots.intersection(test_prots)) == 0, "Protein leakage between train and test in unseen_both!"

    # Sequence leakage check
    assert len(set(sequences[train_idx]).intersection(set(sequences[test_idx]))) == 0, "Sequence leakage between train and test in unseen_both!"
