# DATA CARD: BindingDB Human DTI Binary Benchmark

A standardized Dataset Card following data documentation best practices (Gebru et al., *Datasheets for Datasets*), summarizing the provenance, curation, statistics, feature encodings, and empirical benchmark findings of the curated BindingDB human drug-target interaction dataset.

---

## 1. Dataset Summary

- **Dataset Name:** `bindingdb_human_dti_binary.tsv`
- **Domain:** Computational Chemistry, Molecular Biology, Drug Discovery, Machine Learning
- **Primary Task:** Binary Drug-Target Interaction (DTI) Prediction
- **Source Database:** [BindingDB](https://www.bindingdb.org/)
- **Total Curated Interaction Pairs:** **289,282**
- **Unique Chemical Compounds (Drugs):** **176,283** (canonical SMILES)
- **Unique Target Proteins (UniProt IDs):** **3,253** non-null UniProt IDs (**3,262** target grouping keys when unmapped entries fall back to sequence)
- **Unique Protein Sequences:** **3,594**
- **Class Balance:** **72.03% Positive Binders** (208,359 pairs) / **27.97% Non-Binders** (80,923 pairs)

---

## 2. Data Curation & Filtering Pipeline

The raw BindingDB archive (`BindingDB_All.tsv`, ~9.0 GB) was parsed in chunks using Python (`filter_bindingdb.py` and `curate_dti_binary.py`):

1. **Species Filtering:**
   - Filtered strictly for human targets (`Target Source Organism According to Curator or DataSource == "Homo sapiens"` or SwissProt entry `*_HUMAN`, excluding viral proteins).
2. **Quality & Completeness Constraints:**
   - Retained only entries with a valid, RDKit-parsable SMILES string.
   - Retained only entries with a non-null, unambiguous standard amino acid sequence.
   - Retained only entries with a defined binding affinity ($K_i$ or $K_d$ in nM).
   - Produced `bindingdb_human_ki_kd.tsv` containing **595,804** records.
3. **Deduplication & Conflict Resolution:**
   - Small molecules converted to canonical SMILES via RDKit (`Chem.MolToSmiles(Chem.MolFromSmiles(s))`).
   - Pairs with conflicting binary labels across multiple experimental measurements (measured as both $\le 100\text{ nM}$ and $\ge 10\ \mu\text{M}$) are discarded.
   - For remaining concordant duplicate pairs sharing the same canonical SMILES and target sequence, the first occurrence is retained.
4. **Binary Activity Thresholding:**
   - **Active Binders (Class 1):** Affinity ($K_i$ or $K_d$) $\le 100\text{ nM}$ ($\text{pAffinity} \ge 7.0$).
   - **Inactive / Non-Binders (Class 0):** Affinity ($K_i$ or $K_d$) $\ge 10\ \mu\text{M}$ ($10,000\text{ nM}$, $\text{pAffinity} \le 5.0$).
   - **Intermediate Affinity Discard:** Pairs with affinities between $100\text{ nM}$ and $10\ \mu\text{M}$ were discarded to establish a clean 2-order-of-magnitude activity cliff, preventing label noise near the decision boundary.

```
Raw BindingDB (~9 GB)
       │
       ▼  (Filter human, valid SMILES, sequence, Ki/Kd)
bindingdb_human_ki_kd.tsv (595,804 records)
       │
       ▼  (Canonical SMILES, deduplication, binary thresholding <=100 nM vs >=10 uM)
bindingdb_human_dti_binary.tsv (289,282 binary pairs)
```

---

## 3. Exploratory Data Analysis & Degree Distributions

### Drug Degree (Pairs per Drug)
- **Counting Metric:** Grouped by `canonical_smiles` across all 289,282 interaction pairs.
- **Median:** `1.0` pair per compound (Mean: `1.64`).
- **Singleton Compounds:** **71.39%** of drugs (125,840 out of 176,283) have exactly one interaction pair in the dataset.
- **Distribution Shape:** Extreme power-law distribution. While 71.39% of compounds are tested against only a single target, broad polypharmacology probes and kinase inhibitors (e.g., Staurosporine, Sunitinib, Dasatinib) appear in hundreds of pairs.

### Protein Target Degree (Pairs per Target)
- **Counting Metric:** Evaluated across the **3,253** unique non-null UniProt IDs (`df[df["uniprot_id"].notna()]["uniprot_id"]`), covering 289,255 pairs (27 pairs across 9 unique sequences lack UniProt IDs).
- **Median:** `3.0` pairs per protein (Mean: `88.92`).
- **Singleton Targets:** **36.74%** of UniProt IDs (1,195 out of 3,253) have exactly one pair.
- **Target Grouping Keys (`uniprot_id.fillna(sequence)`):** Across all **3,262** grouping targets, median = `3.0` pairs (Mean: `88.68`).
- **Target Sequences:** Across all **3,594** unique sequences, median = `3.0` pairs (Mean: `80.49`).
- **Distribution Shape:** Highly skewed. Over 100 heavily studied targets (e.g., EGFR, ABL1, CDK2, HER2, GPCRs) have $>500$ measured compound interactions, while niche targets have $\le 3$ pairs.

### Target Label Distribution
- **Counting Metric:** For the **1,053** UniProt IDs with at least 20 measured pairs:
  - **18.04%** of proteins (190 targets) have $>90\%$ positive binders (assay publication bias towards active series).
  - **22.13%** of proteins (233 targets) have $<10\%$ positive binders.
  - Mean fraction labelled 1: `0.484`, median fraction labelled 1: `0.489`.
- **Multi-Sequence Targets:** **134 UniProt IDs (4.12%)** are associated with multiple sequences (isoforms, construct variants, or sequence fragments, with up to 25 distinct sequences for a single UniProt ID). In splits 3 and 4, all sequences sharing a UniProt ID are strictly held on the same side of the partition to prevent isoform leakage.

---

## 4. Benchmark Splitting Schemes

All splits enforce an 80% train / 20% test split, with a 10% validation set carved from training using identical grouping (seed `42`):

| Split | Regime | Grouping Key | Train Pairs | Val Pairs | Test Pairs | % Pos (Test) | Discarded Pairs | Leakage Guarantee |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **Random Pairs** | None (uniform shuffle) | 208,282 | 23,143 | 57,857 | 71.93% | 0 | None (both drugs and targets overlap) |
| **2** | **Unseen Drugs** | `canonical_smiles` | 208,742 | 23,752 | 56,788 | 73.53% | 0 | **0.0%** drug overlap (zero test drugs in train) |
| **3** | **Unseen Proteins** | `uniprot_id` / sequence | 207,196 | 22,287 | 59,799 | 72.06% | 0 | **0.0%** target sequence overlap (zero test targets in train) |
| **4** | **Unseen Both** | Independent 20% drug & 20% target holdout | 154,052 | 1,316 | 11,735 | 73.66% | 122,179 (42.24%) | **0.0%** drug AND **0.0%** target overlap |

> **Note on Split 4:** 122,179 pairs (**42.24%** of the dataset) where only one entity was held out were discarded to ensure the test set represents a pure cold-start interaction space.

---

## 5. Feature Encodings

Precomputed in `features/` to eliminate runtime computational redundancy:

1. **Small Molecule Drugs (ECFP4 Fingerprints):**
   - RDKit Morgan Fingerprint, radius = 2, bit vector size = 2,048.
   - Stored as `uint8` (`features/drug_features.npz` and `.npy`).
   - Shape: `(176283, 2048)`, Memory: 344.30 MB.
2. **Protein Targets (Composition Features):**
   - Amino Acid Composition (AAC): 20 standard amino acid normalized frequencies.
   - Dipeptide Composition (DPC): 400 normalized frequencies of all adjacent pairs ($20 \times 20$).
   - Stored as `float32` (`features/protein_features.npz` and `.npy`).
   - Shape: `(3594, 420)`, Memory: 5.76 MB.
3. **Concatenated Pair Representation:**
   - 2,468 total dimensions ($2048 + 420$). Built dynamically in 20,000-sample batches during model training.

---

## 6. Empirical Benchmark Results

All models were evaluated on the test set of each split using uniform 150,000-sample training sets (`--train-size 150000`, with earlier 40,000-sample baselines archived in `results_40k/`) and 1,000-sample bootstrap 95% confidence intervals:

| Model | Split 1 (Random) ROC-AUC | Split 2 (Unseen Drugs) ROC-AUC | Split 3 (Unseen Proteins) ROC-AUC | Split 4 (Cold-Start) ROC-AUC | Retention Ratio (Split 4 / Split 1) | % Performance Lost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy** | 0.5000 [0.500, 0.500] | 0.5000 [0.500, 0.500] | 0.5000 [0.500, 0.500] | 0.5000 [0.500, 0.500] | 0.0000 | 100.0% |
| **Protein-Only** | 0.8477 [0.844, 0.851] | 0.8419 [0.838, 0.845] | 0.5000 [0.500, 0.500] | 0.5000 [0.500, 0.500] | 0.0000 | 100.0% |
| **Drug-Only** | 0.7995 [0.796, 0.803] | 0.5000 [0.500, 0.500] | 0.7844 [0.781, 0.788] | 0.5000 [0.500, 0.500] | 0.0000 | 100.0% |
| **Nearest-Protein** | 0.5259 [0.521, 0.531] | 0.5282 [0.523, 0.533] | 0.4792 [0.474, 0.484] | 0.5092 [0.499, 0.521] | 0.3565 | 64.35% |
| **RF drug-features-only** | 0.9545 [0.953, 0.956] | 0.8811 [0.878, 0.884] | 0.8659 [0.863, 0.869] | 0.7406 [0.730, 0.751] | 0.5295 | **47.05%** |
| **RF protein-features-only** | 0.8528 [0.849, 0.856] | 0.8469 [0.844, 0.850] | 0.6377 [0.632, 0.643] | 0.6183 [0.607, 0.630] | 0.3352 | 66.48% |
| **Random Forest** | 0.9842 [0.983, 0.985] | 0.9824 [0.981, 0.983] | 0.7453 [0.741, 0.750] | 0.6903 [0.680, 0.701] | 0.3930 | **60.70%** |
| **PyTorch MLP** | 0.9641 [0.962, 0.966] | 0.8991 [0.896, 0.902] | 0.8505 [0.847, 0.854] | 0.7512 [0.742, 0.761] | 0.5413 | **45.87%** |

### Key Diagnostic Observations:
- **Baseline Memorisation at Scale:** On Split 1 with 150k training pairs, the `Drug-Only` baseline leaps from 0.713 (at 40k) to **0.7995**, while `Protein-Only` reaches **0.8477**, confirming that densely sampled entities enable heuristics to achieve deceptively high warm-start scores without modeling interaction physics.
- **Nearest-Protein Limit:** When substituting the positive rate of the nearest homologous training target (by composition cosine similarity), ROC-AUC hovers around **~0.479–0.528**, showing that global sequence homology alone fails to transfer binding selectivity.
- **Feature Modality Ablation (RF):**
  - **RF drug-features-only** (2,048 ECFP4 bits): Achieves 0.9545 on Split 1 and maintains **0.7406** on Split 4 (retaining 52.95% excess AUC). Notably on Split 3 (unseen proteins), it achieves **0.8659**, vastly outperforming full RF (0.7453) by avoiding overfitting to training protein composition.
  - **RF protein-features-only** (420 composition dimensions): Reaches 0.8528 on Split 1 (matching the protein-only baseline) and drops sharply to 0.6377 on Split 3 and 0.6183 on Split 4 (losing 66.48% performance).
- **Neural Network vs. Forest in Cold Start:** At 150k training pairs, Random Forest warm-start ROC-AUC climbs to 0.9842, but drops to 0.6903 on Split 4 (losing **60.7%** of excess performance). In contrast, PyTorch MLP retains **0.7512** on Split 4 (losing **45.9%**), and achieves **0.8505** on Split 3, demonstrating that dense latent representations generalize far better across novel targets.
- **Feature Importance:** Random Forest splits allocate **56.39%** importance to ECFP4 chemical fingerprint bits and **43.61%** to protein sequence composition features.

---

## 7. Known Limitations & Potential Biases

1. **Assay Publication Bias:** The dataset exhibits an overall ~72% positive rate because academic and patent literature rarely reports inactive compounds unless synthesized as part of an active SAR series.
2. **Target Family Skew:** Kinases, GPCRs, and nuclear receptors represent over 60% of interaction pairs, reflecting historical pharma interest.
3. **Compositional Target Representation:** AAC and DPC features capture global amino acid distribution and dipeptide motifs but do not explicitly encode 3D binding pocket geometry or residue ordering.
4. **Conservative Cold-Start Bias:** On Split 4, the highest predicted probability for any inactive pair was $\hat{p} = 0.6150$, while active binders were frequently underpredicted down to $\hat{p} = 0.1550$ (severe false negative skew in novel target families).

---

## 8. File Structure & Reproducibility

- `bindingdb_human_dti_binary.tsv`: Main binary dataset.
- `splits/`: Train, validation, and test indices for splits 1–4.
- `features/`: Precomputed ECFP4 and AAC+DPC features.
- Master pipeline script: `python run_all.py` (guarantees bitwise exact reproduction under random seed `42`).
