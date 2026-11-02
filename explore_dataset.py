"""
Example script to load and inspect the filtered BindingDB human DTI dataset.
Demonstrates:
1. Loading with pandas
2. Inspecting affinity values (Ki, Kd)
3. Parsing SMILES with RDKit
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


def explore():
    dataset_path = "bindingdb_human_ki_kd.tsv"
    print(f"Loading first 10,000 records from {dataset_path}...")
    df = pd.read_csv(dataset_path, sep="\t", nrows=10000, low_memory=False)

    print(f"Loaded shape: {df.shape}")
    print("\nColumns available:")
    for col in df.columns:
        print(f" - {col}")

    print("\nSample DTI pairs:")
    sample = df[["BindingDB Ligand Name", "Ligand SMILES", "Target Name", "Ki (nM)", "Kd (nM)"]].dropna(how="all").head(5)
    print(sample.to_string(index=False))

    # Demonstrate RDKit molecular parsing on the first SMILES
    first_smiles = df["Ligand SMILES"].iloc[0]
    first_target = df["Target Name"].iloc[0]
    mol = Chem.MolFromSmiles(first_smiles)
    if mol:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        print(f"\nRDKit parsed compound for '{first_target}':")
        print(f" - SMILES: {first_smiles[:60]}...")
        print(f" - Molecular Weight: {mw:.2f}")
        print(f" - LogP: {logp:.2f}")
        print(f" - Heavy Atoms: {mol.GetNumHeavyAtoms()}")


if __name__ == "__main__":
    explore()
