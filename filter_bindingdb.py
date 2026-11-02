#!/usr/bin/env python3
"""
Filter BindingDB_All.tsv in chunks to extract human Drug-Target Interaction (DTI) pairs.

Criteria:
1. Organism: Human (Homo sapiens / Human / UniProt *_HUMAN, excluding viral proteins)
2. Has valid Ligand SMILES
3. Has Target Protein Sequence (Chain 1)
4. Has a Ki (nM) or Kd (nM) measurement value
"""

import sys
import os
import time
import argparse
import pandas as pd


# Core columns essential for DTI analysis and modeling
CURATED_COLUMNS = [
    "BindingDB Reactant_set_id",
    "Ligand SMILES",
    "Ligand InChI",
    "Ligand InChI Key",
    "BindingDB MonomerID",
    "BindingDB Ligand Name",
    "Target Name",
    "Target Source Organism According to Curator or DataSource",
    "Ki (nM)",
    "Kd (nM)",
    "IC50 (nM)",
    "EC50 (nM)",
    "pH",
    "Temp (C)",
    "Number of Protein Chains in Target (>1 implies a multichain complex)",
    "BindingDB Target Chain Sequence 1",
    "PDB ID(s) of Target Chain 1",
    "UniProt (SwissProt) Recommended Name of Target Chain 1",
    "UniProt (SwissProt) Entry Name of Target Chain 1",
    "UniProt (SwissProt) Primary ID of Target Chain 1",
    "PubChem CID",
    "ChEMBL ID of Ligand",
    "DrugBank ID of Ligand",
    "Article DOI",
    "PMID",
]


def is_human_target(org_series: pd.Series, uniprot_series: pd.Series) -> pd.Series:
    """Identify human targets while excluding human-infecting viruses."""
    org_clean = org_series.fillna("").astype(str).str.strip().str.lower()
    uni_clean = uniprot_series.fillna("").astype(str).str.strip()

    is_human_org = org_clean.isin(["homo sapiens", "human"])
    is_human_uni = uni_clean.str.endswith("_HUMAN")
    is_virus = org_clean.str.contains(
        r"virus|herpes|sars|hiv|coronavirus|cytomegalo|papilloma", regex=True
    )

    return (is_human_org | is_human_uni) & ~is_virus


def filter_bindingdb(
    input_path: str = "BindingDB_All.tsv",
    output_path: str = "bindingdb_human_ki_kd.tsv",
    chunk_size: int = 50_000,
    keep_all_columns: bool = False,
):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    file_size_bytes = os.path.getsize(input_path)
    file_size_gb = file_size_bytes / (1024**3)
    print(f"Starting chunked filtering of '{input_path}' ({file_size_gb:.2f} GB)...")
    print(f"Chunk size: {chunk_size:,} rows")
    print(f"Output target: '{output_path}'")
    print(f"Keeping all columns: {keep_all_columns}")

    start_time = time.time()
    total_processed = 0
    total_matched = 0
    header_written = False

    # Read TSV in chunks to ensure minimal memory usage
    chunk_iter = pd.read_csv(
        input_path,
        sep="\t",
        chunksize=chunk_size,
        low_memory=False,
        on_bad_lines="skip",
    )

    for i, chunk in enumerate(chunk_iter, 1):
        chunk_len = len(chunk)
        total_processed += chunk_len

        # 1. Human target filter
        org_col = chunk["Target Source Organism According to Curator or DataSource"]
        uni_col = chunk["UniProt (SwissProt) Entry Name of Target Chain 1"]
        human_mask = is_human_target(org_col, uni_col)

        # 2. SMILES filter
        smiles_col = chunk["Ligand SMILES"].fillna("").astype(str).str.strip()
        has_smiles = smiles_col != ""

        # 3. Protein sequence filter
        seq_col = (
            chunk["BindingDB Target Chain Sequence 1"].fillna("").astype(str).str.strip()
        )
        has_seq = seq_col != ""

        # 4. Ki or Kd filter
        ki_col = chunk["Ki (nM)"].fillna("").astype(str).str.strip()
        kd_col = chunk["Kd (nM)"].fillna("").astype(str).str.strip()
        has_ki_kd = (ki_col != "") | (kd_col != "")

        # Combined filter mask
        combined_mask = human_mask & has_smiles & has_seq & has_ki_kd
        matched_chunk = chunk[combined_mask]
        num_matched = len(matched_chunk)
        total_matched += num_matched

        # Select columns to save
        if not keep_all_columns:
            selected_cols = [c for c in CURATED_COLUMNS if c in matched_chunk.columns]
            matched_chunk = matched_chunk[selected_cols]

        # Append to output file
        if num_matched > 0:
            matched_chunk.to_csv(
                output_path,
                sep="\t",
                mode="w" if not header_written else "a",
                index=False,
                header=not header_written,
            )
            header_written = True

        elapsed = time.time() - start_time
        speed = total_processed / elapsed if elapsed > 0 else 0
        if i % 10 == 0 or num_matched > 0:
            print(
                f"Chunk {i:3d}: Processed {total_processed:>9,} rows | "
                f"Matched: {total_matched:>7,} ({total_matched/total_processed*100:5.2f}%) | "
                f"Elapsed: {elapsed:5.1f}s ({speed:6,.0f} rows/s)"
            )

    elapsed = time.time() - start_time
    out_size_mb = os.path.getsize(output_path) / (1024**2) if os.path.exists(output_path) else 0

    print("=" * 60)
    print("Filtering completed successfully!")
    print(f"Total processed: {total_processed:,} rows")
    print(f"Total matched:   {total_matched:,} rows ({total_matched/total_processed*100:.2f}%)")
    print(f"Output saved to: {output_path} ({out_size_mb:.2f} MB)")
    print(f"Total time:      {elapsed:.1f} seconds ({total_processed/elapsed:,.0f} rows/s)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter BindingDB TSV in chunks.")
    parser.add_argument(
        "--input", default="BindingDB_All.tsv", help="Path to input BindingDB TSV file"
    )
    parser.add_argument(
        "--output", default="bindingdb_human_ki_kd.tsv", help="Path to output TSV file"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="Chunk size for reading (default: 50,000)",
    )
    parser.add_argument(
        "--all-cols",
        action="store_true",
        help="Keep all 640 columns instead of curated DTI columns",
    )

    args = parser.parse_args()
    filter_bindingdb(
        input_path=args.input,
        output_path=args.output,
        chunk_size=args.chunk_size,
        keep_all_columns=args.all_cols,
    )
