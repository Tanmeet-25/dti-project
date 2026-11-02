"""
Analyze frequency of drugs and proteins in the curated BindingDB dataset and plot top distributions.
"""

import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def create_plots():
    data_path = "bindingdb_human_dti_binary.tsv"
    print(f"Loading {data_path}...")
    df = pd.read_csv(data_path, sep="\t", low_memory=False)

    n_pairs = len(df)
    n_drugs = df["canonical_smiles"].nunique()
    n_seqs = df["protein_sequence"].nunique()
    n_targets = df["target_name"].nunique()
    n_uniprot = df["uniprot_id"].dropna().nunique()

    print("=" * 50)
    print("DATASET CARDINALITY")
    print("=" * 50)
    print(f"Total Unique Pairs:            {n_pairs:,}")
    print(f"Unique Drugs (SMILES):         {n_drugs:,}")
    print(f"Unique Protein Sequences:      {n_seqs:,}")
    print(f"Unique Target Names:           {n_targets:,}")
    print(f"Unique UniProt IDs:            {n_uniprot:,}")
    print("=" * 50)

    # 1. Top Proteins
    top_proteins = df["target_name"].value_counts().head(12)

    # 2. Top Drugs
    top_smiles = df["canonical_smiles"].value_counts().head(12)
    top_drug_labels = []
    for s in top_smiles.index:
        sub = df[df["canonical_smiles"] == s].iloc[0]
        raw_name = str(sub["ligand_name"]) if pd.notna(sub["ligand_name"]) else ""
        cid = int(sub["pubchem_cid"]) if pd.notna(sub["pubchem_cid"]) else None

        # Determine best readable label
        display_name = ""
        if raw_name:
            candidates = [p.strip() for p in raw_name.split("::") if p.strip()]
            short_candidates = [c for c in candidates if len(c) <= 28 and not c.startswith("Scaffold")]
            if short_candidates:
                display_name = short_candidates[-1]
            else:
                display_name = candidates[0][:25] + "..." if len(candidates[0]) > 25 else candidates[0]

        if not display_name and cid:
            display_name = f"PubChem CID {cid}"
        elif not display_name:
            display_name = s[:20] + "..."

        if cid and str(cid) not in display_name and len(display_name) < 22:
            display_name += f" (CID {cid})"

        top_drug_labels.append(display_name)

    # Styling settings
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), dpi=300)
    fig.patch.set_facecolor("#FAFAFC")

    # Panel A: Top Proteins
    ax1 = axes[0]
    ax1.set_facecolor("#FFFFFF")
    prot_names = top_proteins.index.tolist()[::-1]
    prot_counts = top_proteins.values.tolist()[::-1]

    bars1 = ax1.barh(range(len(prot_names)), prot_counts, color="#2B6CB0", edgecolor="#1A365D", height=0.68, alpha=0.9)
    ax1.set_yticks(range(len(prot_names)))
    ax1.set_yticklabels(prot_names, fontsize=10.5, fontweight="500", color="#2D3748")
    ax1.set_xlabel("Number of Interaction Pairs", fontsize=11, fontweight="600", color="#2D3748", labelpad=10)
    ax1.set_title("Top 12 Most Frequent Human Target Proteins", fontsize=13, fontweight="700", color="#1A202C", pad=12)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
    ax1.set_xlim(0, max(prot_counts) * 1.15)
    ax1.grid(axis="x", linestyle="--", alpha=0.45, color="#CBD5E0")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color("#E2E8F0")
    ax1.spines["bottom"].set_color("#CBD5E0")

    # Add count labels
    for bar in bars1:
        w = bar.get_width()
        ax1.text(w + max(prot_counts) * 0.015, bar.get_y() + bar.get_height() / 2, f"{int(w):,}",
                 ha="left", va="center", fontsize=9.5, fontweight="600", color="#2B6CB0")

    # Panel B: Top Drugs
    ax2 = axes[1]
    ax2.set_facecolor("#FFFFFF")
    drug_names = top_drug_labels[::-1]
    drug_counts = top_smiles.values.tolist()[::-1]

    bars2 = ax2.barh(range(len(drug_names)), drug_counts, color="#DD6B20", edgecolor="#9C4221", height=0.68, alpha=0.9)
    ax2.set_yticks(range(len(drug_names)))
    ax2.set_yticklabels(drug_names, fontsize=10, fontweight="500", color="#2D3748")
    ax2.set_xlabel("Number of Target Proteins Tested", fontsize=11, fontweight="600", color="#2D3748", labelpad=10)
    ax2.set_title("Top 12 Most Frequent Compounds / Drugs", fontsize=13, fontweight="700", color="#1A202C", pad=12)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
    ax2.set_xlim(0, max(drug_counts) * 1.15)
    ax2.grid(axis="x", linestyle="--", alpha=0.45, color="#CBD5E0")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color("#E2E8F0")
    ax2.spines["bottom"].set_color("#CBD5E0")

    # Add count labels
    for bar in bars2:
        w = bar.get_width()
        ax2.text(w + max(drug_counts) * 0.015, bar.get_y() + bar.get_height() / 2, f"{int(w):,}",
                 ha="left", va="center", fontsize=9.5, fontweight="600", color="#C05621")

    fig.suptitle(f"BindingDB Human DTI Dataset Overview ({n_pairs:,} Pairs | {n_drugs:,} Drugs | {n_targets:,} Targets)",
                 fontsize=14.5, fontweight="700", color="#1A202C", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_file = "most_frequent_drugs_and_proteins.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to '{out_file}'")


if __name__ == "__main__":
    create_plots()
