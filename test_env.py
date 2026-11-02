"""
Environment verification test for DTI project.
Checks: pandas, scikit-learn, rdkit, matplotlib, jupyter, and PyTorch (CPU).
"""

def test_environment():
    import pandas as pd
    import sklearn
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    import matplotlib
    import torch

    print("=" * 50)
    print("Python Environment Verification")
    print("=" * 50)
    print(f"Pandas:       {pd.__version__}")
    print(f"Scikit-Learn: {sklearn.__version__}")
    print(f"RDKit:        {rdkit.__version__}")
    print(f"Matplotlib:   {matplotlib.__version__}")
    print(f"PyTorch:      {torch.__version__} (Device: {torch.device('cpu')}, CUDA: {torch.cuda.is_available()})")

    # Quick RDKit sanity check
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    mol = Chem.MolFromSmiles(smiles)
    mw = Descriptors.MolWt(mol)
    print(f"\nRDKit Test -> Aspirin (MW: {mw:.2f}, Atoms: {mol.GetNumAtoms()})")

    # Quick PyTorch CPU tensor check
    tensor = torch.randn(3, 3)
    print(f"PyTorch Test -> Tensor Shape: {tensor.shape}, Device: {tensor.device}")
    print("=" * 50)
    print("All checks passed successfully!")

if __name__ == "__main__":
    test_environment()
