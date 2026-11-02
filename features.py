#!/usr/bin/env python3
"""
Feature Extraction for Drug-Target Interaction (DTI):
1. Drug Features: ECFP4 fingerprints (Morgan radius 2, 2048 bits) as uint8 for unique drugs.
2. Protein Features: Amino Acid Composition (20) + Dipeptide Composition (400) as float32 for unique proteins.
3. ID Lookup Tables: Fast index mapping between SMILES/sequences and their feature row indices.
Saves precomputed arrays to features/ folder (.npy and .npz). Does NOT build the pair matrix.
"""

import os
import sys
import time
import json
import argparse
from itertools import product
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


# 20 standard amino acids in alphabetical order
STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: i for i, aa in enumerate(STANDARD_AMINO_ACIDS)}
DIPEPTIDES = [a1 + a2 for a1, a2 in product(STANDARD_AMINO_ACIDS, STANDARD_AMINO_ACIDS)]
DIPEPTIDE_TO_INDEX = {dp: i for i, dp in enumerate(DIPEPTIDES)}


def compute_drug_features(unique_smiles: list[str]) -> np.ndarray:
    """
    Compute ECFP4 fingerprints (Morgan radius 2, 2048 bits) as uint8 for each SMILES.
    """
    n_drugs = len(unique_smiles)
    fp_size = 2048
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)

    features = np.zeros((n_drugs, fp_size), dtype=np.uint8)
    for i, smiles in enumerate(unique_smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            features[i] = morgan_gen.GetFingerprintAsNumPy(mol)
        if (i + 1) % 50000 == 0 or (i + 1) == n_drugs:
            print(f"   [Drugs] Processed {i + 1:,} / {n_drugs:,} compounds...")

    return features


def compute_protein_features(unique_sequences: list[str]) -> np.ndarray:
    """
    Compute AAC (20) and DPC (400) features as float32 for each protein sequence.
    Total feature dimensions: 420.
    """
    n_prots = len(unique_sequences)
    feat_dim = 20 + 400
    features = np.zeros((n_prots, feat_dim), dtype=np.float32)

    for i, seq in enumerate(unique_sequences):
        clean_seq = seq.strip().upper()
        L = len(clean_seq)
        if L == 0:
            continue

        # 1. Amino Acid Composition (AAC - 20)
        for aa in clean_seq:
            idx = AA_TO_INDEX.get(aa)
            if idx is not None:
                features[i, idx] += 1.0
        features[i, :20] /= L

        # 2. Dipeptide Composition (DPC - 400)
        if L > 1:
            for j in range(L - 1):
                dp = clean_seq[j : j + 2]
                dp_idx = DIPEPTIDE_TO_INDEX.get(dp)
                if dp_idx is not None:
                    features[i, 20 + dp_idx] += 1.0
            features[i, 20:] /= (L - 1)

    return features


def extract_and_save_features(
    input_path: str = "bindingdb_human_dti_binary.tsv",
    out_dir: str = "features",
):
    start_total = time.time()
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 65)
    print("DTI Feature Precomputation Pipeline")
    print("=" * 65)

    if not os.path.exists(input_path):
        print(f"Error: Dataset '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"1. Loading dataset from '{input_path}'...")
    df = pd.read_csv(input_path, sep="\t", usecols=["canonical_smiles", "protein_sequence"], low_memory=False)

    unique_drugs = df["canonical_smiles"].unique().tolist()
    unique_proteins = df["protein_sequence"].unique().tolist()

    print(f"   Unique Drugs (canonical_smiles): {len(unique_drugs):,}")
    print(f"   Unique Proteins (sequences):      {len(unique_proteins):,}")

    # 1. Compute Drug Features
    print(f"\n2. Computing ECFP4 fingerprints (radius 2, 2048 bits, uint8)...")
    t0_drug = time.time()
    drug_features = compute_drug_features(unique_drugs)
    t_drug = time.time() - t0_drug
    print(f"   Done in {t_drug:.2f}s ({len(unique_drugs) / t_drug:,.0f} drugs/s)")
    print(f"   Drug features shape: {drug_features.shape}, dtype: {drug_features.dtype}")

    # 2. Compute Protein Features
    print(f"\n3. Computing AAC (20) + DPC (400) protein features (float32)...")
    t0_prot = time.time()
    protein_features = compute_protein_features(unique_proteins)
    t_prot = time.time() - t0_prot
    print(f"   Done in {t_prot:.2f}s ({len(unique_proteins) / t_prot:,.0f} proteins/s)")
    print(f"   Protein features shape: {protein_features.shape}, dtype: {protein_features.dtype}")

    # 3. Create ID Lookup Tables
    print(f"\n4. Constructing ID lookup tables...")
    drug_smiles_arr = np.array(unique_drugs, dtype=object)
    protein_seq_arr = np.array(unique_proteins, dtype=object)

    drug_to_idx = {s: i for i, s in enumerate(unique_drugs)}
    protein_to_idx = {p: i for i, p in enumerate(unique_proteins)}

    # 4. Save to features/ folder
    print(f"\n5. Saving features and lookup tables to '{out_dir}/'...")
    t0_save = time.time()

    drug_feat_path = os.path.join(out_dir, "drug_features.npy")
    prot_feat_path = os.path.join(out_dir, "protein_features.npy")
    drug_smiles_path = os.path.join(out_dir, "drug_smiles.npy")
    prot_seq_path = os.path.join(out_dir, "protein_sequences.npy")
    bundle_path = os.path.join(out_dir, "dti_features.npz")
    drug_map_path = os.path.join(out_dir, "drug_to_idx.json")
    prot_map_path = os.path.join(out_dir, "protein_to_idx.json")

    np.save(drug_feat_path, drug_features)
    np.save(prot_feat_path, protein_features)
    np.save(drug_smiles_path, drug_smiles_arr)
    np.save(prot_seq_path, protein_seq_arr)

    # Also save compressed NPZ bundle for portable loading
    np.savez_compressed(
        bundle_path,
        drug_features=drug_features,
        protein_features=protein_features,
        drug_smiles=drug_smiles_arr,
        protein_sequences=protein_seq_arr,
    )

    with open(drug_map_path, "w", encoding="utf-8") as f:
        json.dump(drug_to_idx, f)
    with open(prot_map_path, "w", encoding="utf-8") as f:
        json.dump(protein_to_idx, f)

    t_save = time.time() - t0_save
    total_time = time.time() - start_total

    # File size inspection
    saved_files = [
        ("Drug Features (.npy)", drug_feat_path, drug_features.shape, drug_features.dtype),
        ("Protein Features (.npy)", prot_feat_path, protein_features.shape, protein_features.dtype),
        ("Drug SMILES Lookup (.npy)", drug_smiles_path, drug_smiles_arr.shape, drug_smiles_arr.dtype),
        ("Protein Sequences Lookup (.npy)", prot_seq_path, protein_seq_arr.shape, protein_seq_arr.dtype),
        ("Compressed Bundle (.npz)", bundle_path, "bundle (all arrays)", "-"),
        ("Drug Index Map (.json)", drug_map_path, f"{len(drug_to_idx):,} keys", "-"),
        ("Protein Index Map (.json)", prot_map_path, f"{len(protein_to_idx):,} keys", "-"),
    ]

    print("\n" + "=" * 65)
    print("PRECOMPUTED FEATURES SUMMARY")
    print("=" * 65)
    print(f"{'Artifact':<32} {'Shape / Keys':<24} {'Size':>10}")
    print("-" * 65)
    total_bytes = 0
    for name, path, shape, _ in saved_files:
        if os.path.exists(path):
            sz_bytes = os.path.getsize(path)
            total_bytes += sz_bytes
            sz_str = f"{sz_bytes / (1024**2):.2f} MB" if sz_bytes > 1024**2 else f"{sz_bytes / 1024:.2f} KB"
            print(f"{name:<32} {str(shape):<24} {sz_str:>10}")

    print("-" * 65)
    print(f"Total Disk Footprint: {total_bytes / (1024**2):.2f} MB")
    print(f"Drug Feature Time:    {t_drug:.2f} s")
    print(f"Protein Feature Time: {t_prot:.2f} s")
    print(f"File Saving Time:     {t_save:.2f} s")
    print(f"Total Execution Time: {total_time:.2f} s")
    print("=" * 65)


def load_features(features_dir: str = "features"):
    """
    Convenience function to load precomputed features and lookup tables.
    Returns:
        drug_features: np.ndarray (N_drugs, 2048) uint8
        protein_features: np.ndarray (N_proteins, 420) float32
        drug_to_idx: dict[str, int]
        protein_to_idx: dict[str, int]
    """
    drug_feat_path = os.path.join(features_dir, "drug_features.npy")
    prot_feat_path = os.path.join(features_dir, "protein_features.npy")
    drug_map_path = os.path.join(features_dir, "drug_to_idx.json")
    prot_map_path = os.path.join(features_dir, "protein_to_idx.json")

    drug_features = np.load(drug_feat_path, mmap_mode="r")
    protein_features = np.load(prot_feat_path, mmap_mode="r")

    with open(drug_map_path, "r", encoding="utf-8") as f:
        drug_to_idx = json.load(f)
    with open(prot_map_path, "r", encoding="utf-8") as f:
        protein_to_idx = json.load(f)

    return drug_features, protein_features, drug_to_idx, protein_to_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and save drug and protein features.")
    parser.add_argument("--input", default="bindingdb_human_dti_binary.tsv", help="Input dataset TSV")
    parser.add_argument("--outdir", default="features", help="Output directory for features")
    args = parser.parse_args()

    extract_and_save_features(input_path=args.input, out_dir=args.outdir)
