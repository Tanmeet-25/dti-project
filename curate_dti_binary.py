#!/usr/bin/env python3
"""
Curate BindingDB Human DTI dataset for binary classification:
1. Remove broken SMILES using RDKit validation
2. Canonicalize valid SMILES
3. Label strong binders (<=100 nM) as 1 and weak binders (>=10 uM / 10,000 nM) as 0
4. Drop intermediate affinity values (100 nM < value < 10,000 nM)
5. Remove duplicate and conflicting (SMILES, sequence) pairs
6. Save curated dataset to TSV
"""

import sys
import os
import re
import time
import argparse
import pandas as pd
from rdkit import Chem


def parse_affinity_string(val):
    """Parse numeric value and inequality operator from affinity string."""
    if pd.isna(val):
        return None, None
    s = str(val).strip()
    if not s:
        return None, None
    m = re.match(r"^([<>~]?)\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)$", s)
    if not m:
        return None, None
    return m.group(1), float(m.group(2))


def get_binary_label(row):
    """
    Classify record:
      - 1: Strong binder (<= 100 nM)
      - 0: Weak binder (>= 10,000 nM / 10 uM)
      - -1: Intermediate / uncertain (to be dropped)
    """
    for col in ["Ki (nM)", "Kd (nM)"]:
        op, num = parse_affinity_string(row[col])
        if num is not None:
            if op in ("", "="):
                if num <= 100.0:
                    return 1, num, col.split()[0]
                elif num >= 10000.0:
                    return 0, num, col.split()[0]
                else:
                    return -1, num, col.split()[0]
            elif op == ">":
                if num >= 10000.0:
                    return 0, num, col.split()[0]
                else:
                    return -1, num, col.split()[0]
            elif op == "<":
                if num <= 100.0:
                    return 1, num, col.split()[0]
                else:
                    return -1, num, col.split()[0]
    return -1, None, None


def curate_dataset(
    input_tsv: str = "bindingdb_human_ki_kd.tsv",
    output_tsv: str = "bindingdb_human_dti_binary.tsv",
):
    print("=" * 65)
    print("BindingDB Human DTI Binary Dataset Curation")
    print("=" * 65)
    start_time = time.time()

    if not os.path.exists(input_tsv):
        print(f"Error: {input_tsv} does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"1. Loading filtered BindingDB data from '{input_tsv}'...")
    df = pd.read_csv(input_tsv, sep="\t", low_memory=False)
    total_raw = len(df)
    print(f"   Total input records: {total_raw:,}")

    # 1. SMILES Validation & Canonicalization
    print("2. Validating and canonicalizing SMILES with RDKit...")
    unique_smiles = df["Ligand SMILES"].dropna().unique()
    valid_smiles_map = {}
    broken_smiles_count = 0

    for s in unique_smiles:
        try:
            m = Chem.MolFromSmiles(s)
            if m is not None:
                valid_smiles_map[s] = Chem.MolToSmiles(m)
            else:
                broken_smiles_count += 1
        except Exception:
            broken_smiles_count += 1

    df["canonical_smiles"] = df["Ligand SMILES"].map(valid_smiles_map)
    df_valid_smiles = df[df["canonical_smiles"].notna()].copy()
    dropped_broken = total_raw - len(df_valid_smiles)
    print(f"   Unique SMILES evaluated: {len(unique_smiles):,}")
    print(f"   Broken SMILES structures: {broken_smiles_count:,}")
    print(f"   Rows dropped due to broken SMILES: {dropped_broken:,}")
    print(f"   Remaining valid rows: {len(df_valid_smiles):,}")

    # 2. Binary Label Assignment
    print("3. Assigning binary affinity labels (1: <=100 nM, 0: >=10 uM)...")
    results = [get_binary_label(row) for _, row in df_valid_smiles.iterrows()]
    labels, aff_vals, aff_types = zip(*results)

    df_valid_smiles["label"] = labels
    df_valid_smiles["affinity_nM"] = aff_vals
    df_valid_smiles["affinity_type"] = aff_types

    # Drop intermediate values
    labeled_df = df_valid_smiles[df_valid_smiles["label"].isin([0, 1])].copy()
    dropped_between = (df_valid_smiles["label"] == -1).sum()
    print(f"   Intermediate/in-between values dropped: {dropped_between:,}")
    print(f"   Labeled rows before pair deduplication: {len(labeled_df):,}")
    print(f"     - Class 1 (Strong binders): {(labeled_df['label'] == 1).sum():,}")
    print(f"     - Class 0 (Weak binders):   {(labeled_df['label'] == 0).sum():,}")

    # 3. Deduplicate Drug-Target Pairs
    print("4. Deduplicating (canonical_smiles, sequence) pairs...")
    pair_cols = ["canonical_smiles", "BindingDB Target Chain Sequence 1"]

    # Check consistency of labels across duplicates
    pair_label_stats = labeled_df.groupby(pair_cols)["label"].agg(["nunique", "first"])
    conflicting_pairs = pair_label_stats[pair_label_stats["nunique"] > 1]
    concordant_pairs = pair_label_stats[pair_label_stats["nunique"] == 1]

    print(f"   Total unique pairs identified: {len(pair_label_stats):,}")
    print(f"   Conflicting pairs (measured as both 0 and 1): {len(conflicting_pairs):,} (dropped)")
    print(f"   Concordant pairs retained: {len(concordant_pairs):,}")

    # Remove conflicting pairs from labeled dataset
    valid_pairs_set = set(concordant_pairs.index)
    labeled_df["pair_key"] = list(zip(labeled_df["canonical_smiles"], labeled_df["BindingDB Target Chain Sequence 1"]))
    deduped_df = labeled_df[labeled_df["pair_key"].isin(valid_pairs_set)].drop_duplicates(subset=pair_cols, keep="first").copy()
    deduped_df.drop(columns=["pair_key"], inplace=True)

    # Reorder and rename key columns
    output_columns = [
        "canonical_smiles",
        "BindingDB Target Chain Sequence 1",
        "label",
        "affinity_nM",
        "affinity_type",
        "Target Name",
        "UniProt (SwissProt) Primary ID of Target Chain 1",
        "UniProt (SwissProt) Entry Name of Target Chain 1",
        "BindingDB Ligand Name",
        "BindingDB MonomerID",
        "PubChem CID",
        "ChEMBL ID of Ligand",
        "DrugBank ID of Ligand",
        "PDB ID(s) of Target Chain 1",
        "Article DOI",
        "PMID",
    ]
    avail_cols = [c for c in output_columns if c in deduped_df.columns]
    final_df = deduped_df[avail_cols].rename(
        columns={
            "BindingDB Target Chain Sequence 1": "protein_sequence",
            "Target Name": "target_name",
            "UniProt (SwissProt) Primary ID of Target Chain 1": "uniprot_id",
            "UniProt (SwissProt) Entry Name of Target Chain 1": "uniprot_entry",
            "BindingDB Ligand Name": "ligand_name",
            "BindingDB MonomerID": "bindingdb_monomer_id",
            "PubChem CID": "pubchem_cid",
            "ChEMBL ID of Ligand": "chembl_id",
            "DrugBank ID of Ligand": "drugbank_id",
            "PDB ID(s) of Target Chain 1": "pdb_ids",
            "Article DOI": "doi",
            "PMID": "pmid",
        }
    )

    print(f"5. Saving clean dataset to '{output_tsv}'...")
    final_df.to_csv(output_tsv, sep="\t", index=False)
    file_size_mb = os.path.getsize(output_tsv) / (1024**2)

    c1 = (final_df["label"] == 1).sum()
    c0 = (final_df["label"] == 0).sum()
    total_final = len(final_df)
    elapsed = time.time() - start_time

    print("\n" + "=" * 65)
    print("FINAL CLASS DISTRIBUTION & RESULTS")
    print("=" * 65)
    print(f"Output File:     {output_tsv} ({file_size_mb:.2f} MB)")
    print(f"Total Pairs:     {total_final:,}")
    print(f"Class 1 (Strong): {c1:,} ({c1/total_final*100:.2f}%)  [<= 100 nM]")
    print(f"Class 0 (Weak):   {c0:,} ({c0/total_final*100:.2f}%)  [>= 10 uM]")
    print(f"Execution Time:  {elapsed:.1f} seconds")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curate BindingDB DTI dataset for binary classification.")
    parser.add_argument("--input", default="bindingdb_human_ki_kd.tsv", help="Input filtered TSV")
    parser.add_argument("--output", default="bindingdb_human_dti_binary.tsv", help="Output curated TSV")
    args = parser.parse_args()
    curate_dataset(input_tsv=args.input, output_tsv=args.output)
