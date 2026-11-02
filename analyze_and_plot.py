"""
Analysis and plotting script for bindingdb_human_dti_binary.tsv:
1. Log-scale histograms of pairs per drug and pairs per protein (UniProt ID),
   with median and % of drugs with only one pair.
2. Fraction labelled 1 for proteins with >= 20 pairs, histogram,
   and count of proteins >90% or <10%.
3. UniProt IDs with more than one sequence count.
Saves all plots in the results/ folder and copies to artifact directory.
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def run_analysis():
    input_file = "bindingdb_human_dti_binary.tsv"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    print(f"Loading '{input_file}'...")
    df = pd.read_csv(input_file, sep="\t", low_memory=False)
    total_pairs = len(df)

    print("=" * 65)
    print("ANALYSIS RESULTS")
    print("=" * 65)

    # -------------------------------------------------------------
    # (1) Pairs per drug and Pairs per protein (UniProt ID)
    # -------------------------------------------------------------
    drug_counts = df["canonical_smiles"].value_counts()
    median_pairs_per_drug = float(drug_counts.median())
    single_pair_drugs = int((drug_counts == 1).sum())
    total_drugs = len(drug_counts)
    pct_single_pair_drugs = (single_pair_drugs / total_drugs) * 100.0

    df_prot = df[df["uniprot_id"].notna()]
    prot_counts = df_prot["uniprot_id"].value_counts()
    median_pairs_per_prot = float(prot_counts.median())
    total_prots = len(prot_counts)

    print(f"(1) Pairs per Drug:")
    print(f"    - Total unique drugs:                  {total_drugs:,}")
    print(f"    - Median pairs per drug:               {median_pairs_per_drug:.1f}")
    print(f"    - Mean pairs per drug:                 {drug_counts.mean():.2f}")
    print(f"    - Drugs with only one pair:            {single_pair_drugs:,} ({pct_single_pair_drugs:.2f}%)")
    print(f"    Pairs per Protein (UniProt ID):")
    print(f"    - Total unique UniProt IDs:            {total_prots:,}")
    print(f"    - Median pairs per protein:            {median_pairs_per_prot:.1f}")
    print(f"    - Mean pairs per protein:              {prot_counts.mean():.2f}")

    # -------------------------------------------------------------
    # (2) Fraction labelled 1 for proteins with >= 20 pairs
    # -------------------------------------------------------------
    prot_stats = df_prot.groupby("uniprot_id").agg(
        total_pairs=("label", "count"),
        class_1=("label", lambda s: (s == 1).sum()),
        target_name=("target_name", "first"),
    )
    prot_ge20 = prot_stats[prot_stats["total_pairs"] >= 20].copy()
    prot_ge20["frac_1"] = prot_ge20["class_1"] / prot_ge20["total_pairs"]

    n_ge20 = len(prot_ge20)
    above_90 = int((prot_ge20["frac_1"] > 0.90).sum())
    below_10 = int((prot_ge20["frac_1"] < 0.10).sum())
    ge_90 = int((prot_ge20["frac_1"] >= 0.90).sum())
    le_10 = int((prot_ge20["frac_1"] <= 0.10).sum())

    print(f"\n(2) Fraction Labelled 1 (Proteins with >= 20 pairs):")
    print(f"    - Total qualifying proteins (>=20):    {n_ge20:,}")
    print(f"    - Proteins above 90% labelled 1 (>0.90): {above_90:,} ({above_90/n_ge20*100:.2f}%) [>=0.90: {ge_90:,}]")
    print(f"    - Proteins below 10% labelled 1 (<0.10): {below_10:,} ({below_10/n_ge20*100:.2f}%) [<=0.10: {le_10:,}]")
    print(f"    - Mean fraction labelled 1:             {prot_ge20['frac_1'].mean():.3f}")
    print(f"    - Median fraction labelled 1:           {prot_ge20['frac_1'].median():.3f}")

    # -------------------------------------------------------------
    # (3) UniProt IDs with more than one sequence
    # -------------------------------------------------------------
    seq_counts = df_prot.groupby("uniprot_id")["protein_sequence"].nunique()
    multi_seq_unis = int((seq_counts > 1).sum())
    total_uni_evaluated = len(seq_counts)
    pct_multi_seq = (multi_seq_unis / total_uni_evaluated) * 100.0

    print(f"\n(3) Sequence Diversity per UniProt ID:")
    print(f"    - UniProt IDs with > 1 sequence:       {multi_seq_unis:,} out of {total_uni_evaluated:,} ({pct_multi_seq:.2f}%)")
    print(f"    - Max sequences for a single UniProt:  {seq_counts.max()}")
    print("=" * 65)

    # =============================================================
    # PLOTTING
    # =============================================================
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot 1: Pairs per Drug & Pairs per Protein (Log Scale Histograms)
    fig1, (ax_drug, ax_prot) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    fig1.patch.set_facecolor("#FAFAFC")

    # Drug histogram (log bins & log count)
    ax_drug.set_facecolor("#FFFFFF")
    drug_bins = np.logspace(0, np.log10(drug_counts.max()), 35)
    ax_drug.hist(drug_counts.values, bins=drug_bins, color="#DD6B20", edgecolor="#9C4221", alpha=0.85)
    ax_drug.set_xscale("log")
    ax_drug.set_yscale("log")
    ax_drug.axvline(median_pairs_per_drug, color="#C53030", linestyle="--", linewidth=1.8, label=f"Median = {median_pairs_per_drug:.0f}")
    ax_drug.set_xlabel("Pairs per Drug (Log Scale)", fontsize=11, fontweight="600", color="#2D3748")
    ax_drug.set_ylabel("Number of Drugs (Log Scale)", fontsize=11, fontweight="600", color="#2D3748")
    ax_drug.set_title("Distribution of Pairs per Drug", fontsize=12.5, fontweight="700", color="#1A202C", pad=10)
    ax_drug.grid(True, which="both", linestyle="--", alpha=0.35, color="#CBD5E0")
    ax_drug.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0", loc="upper right", fontsize=10)

    # Annotation box for drug stats
    ax_drug.text(
        0.05, 0.22,
        f"Total Drugs: {total_drugs:,}\n"
        f"Median Pairs: {median_pairs_per_drug:.0f}\n"
        f"Single-pair Drugs: {pct_single_pair_drugs:.1f}%\n"
        f"({single_pair_drugs:,} drugs)",
        transform=ax_drug.transAxes,
        fontsize=9.5,
        fontweight="500",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFFAF0", edgecolor="#FBD38D", alpha=0.95),
        color="#7B341E"
    )

    # Protein histogram (log bins & log count)
    ax_prot.set_facecolor("#FFFFFF")
    prot_bins = np.logspace(0, np.log10(prot_counts.max()), 35)
    ax_prot.hist(prot_counts.values, bins=prot_bins, color="#2B6CB0", edgecolor="#1A365D", alpha=0.85)
    ax_prot.set_xscale("log")
    ax_prot.set_yscale("log")
    ax_prot.axvline(median_pairs_per_prot, color="#C53030", linestyle="--", linewidth=1.8, label=f"Median = {median_pairs_per_prot:.0f}")
    ax_prot.set_xlabel("Pairs per Protein (UniProt ID) (Log Scale)", fontsize=11, fontweight="600", color="#2D3748")
    ax_prot.set_ylabel("Number of Proteins (Log Scale)", fontsize=11, fontweight="600", color="#2D3748")
    ax_prot.set_title("Distribution of Pairs per Protein (UniProt ID)", fontsize=12.5, fontweight="700", color="#1A202C", pad=10)
    ax_prot.grid(True, which="both", linestyle="--", alpha=0.35, color="#CBD5E0")
    ax_prot.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0", loc="upper right", fontsize=10)

    # Annotation box for protein stats
    ax_prot.text(
        0.05, 0.22,
        f"Total UniProt IDs: {total_prots:,}\n"
        f"Median Pairs: {median_pairs_per_prot:.0f}\n"
        f"Mean Pairs: {prot_counts.mean():.1f}\n"
        f"Max Pairs: {prot_counts.max():,} (CAH2)",
        transform=ax_prot.transAxes,
        fontsize=9.5,
        fontweight="500",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#EBF8FF", edgecolor="#BEE3F8", alpha=0.95),
        color="#2A4365"
    )

    fig1.suptitle("Log-Scale Distributions of Interaction Pairs per Drug and Protein",
                  fontsize=14, fontweight="700", color="#1A202C", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p1_path = os.path.join(results_dir, "pairs_per_drug_and_protein_histograms.png")
    fig1.savefig(p1_path, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {p1_path}")

    # Plot 2: Fraction Labelled 1 (Proteins with >= 20 pairs)
    fig2, ax_frac = plt.subplots(figsize=(9, 6), dpi=300)
    fig2.patch.set_facecolor("#FAFAFC")
    ax_frac.set_facecolor("#FFFFFF")

    bins = np.linspace(0, 1, 21)
    counts, edges, patches = ax_frac.hist(
        prot_ge20["frac_1"], bins=bins, color="#319795", edgecolor="#234E52", alpha=0.88, rwidth=0.92
    )

    # Highlight <10% (weak-biased) and >90% (strong-biased)
    for i, p in enumerate(patches):
        if edges[i + 1] <= 0.101:
            p.set_facecolor("#E53E3E")  # Red for weak binder bias
            p.set_edgecolor("#9B2C2C")
        elif edges[i] >= 0.899:
            p.set_facecolor("#38A169")  # Green for strong binder bias
            p.set_edgecolor("#22543D")

    ax_frac.set_xlabel("Fraction of Pairs Labelled 1 (Strong Binders, <=100 nM)", fontsize=11, fontweight="600", color="#2D3748")
    ax_frac.set_ylabel("Number of Proteins (UniProt IDs)", fontsize=11, fontweight="600", color="#2D3748")
    ax_frac.set_title("Distribution of Fraction Labelled 1 per Protein (Proteins with >= 20 Pairs)",
                      fontsize=13, fontweight="700", color="#1A202C", pad=12)
    ax_frac.grid(axis="y", linestyle="--", alpha=0.45, color="#CBD5E0")
    ax_frac.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    # Add custom legend handles
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#E53E3E", edgecolor="#9B2C2C", label=f"< 10% Strong (N = {below_10})"),
        Patch(facecolor="#319795", edgecolor="#234E52", label=f"10% - 90% (N = {n_ge20 - above_90 - below_10})"),
        Patch(facecolor="#38A169", edgecolor="#22543D", label=f"> 90% Strong (N = {above_90})"),
    ]
    ax_frac.legend(handles=legend_elements, frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0", loc="upper center", fontsize=10)

    # Summary stat box
    ax_frac.text(
        0.03, 0.75,
        f"Total Proteins (>=20 pairs): {n_ge20:,}\n"
        f"Above 90% labelled 1: {above_90:,} ({above_90/n_ge20*100:.1f}%)\n"
        f"Below 10% labelled 1: {below_10:,} ({below_10/n_ge20*100:.1f}%)\n"
        f"Mean fraction 1: {prot_ge20['frac_1'].mean():.2f}",
        transform=ax_frac.transAxes,
        fontsize=9.5,
        fontweight="500",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F7FAFC", edgecolor="#E2E8F0", alpha=0.95),
        color="#2D3748"
    )

    plt.tight_layout()
    p2_path = os.path.join(results_dir, "protein_fraction_labelled_1_histogram.png")
    fig2.savefig(p2_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {p2_path}")


if __name__ == "__main__":
    run_analysis()
