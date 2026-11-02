# Drug-Target Interaction (DTI) Benchmark: Diagnosing Memorisation vs. Generalisation

A rigorous, end-to-end benchmarking framework for Drug-Target Interaction (DTI) prediction on human protein targets from BindingDB. This project evaluates baseline heuristics and modern machine learning models across four distinct generalization regimes to expose entity memorisation, quantify domain-shift degradation, and measure performance decay as a function of chemical and sequence similarity.

**Live site:** https://tanmeet-25.github.io/dti-project/
---

## Table of Contents
- [Project Goal](#project-goal)
- [Dataset](#dataset)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Interactive Streamlit Dashboard](#interactive-streamlit-dashboard)
- [The Four Generalization Regimes](#the-four-generalization-regimes)
- [Models Evaluated](#models-evaluated)
- [Benchmark Results (150k Training Pairs)](#benchmark-results-150k-training-pairs)
- [Key Result Files & Directory Layout](#key-result-files--directory-layout)

---

## Project Goal
Standard machine learning benchmarks in drug discovery frequently rely on random train/test splits. Because active compounds and popular therapeutic targets (e.g., kinases, GPCRs) are heavily sampled, random splitting yields pairs where both the drug and protein are well-represented in the training set. Models can achieve seemingly stellar ROC-AUC scores (>0.95) simply by memorising target or ligand baselines rather than learning biophysical interaction rules.

The objective of this project is to:
1. **Disentangle interaction learning from entity memorisation** by comparing warm-start evaluation (random pairs) against three strict cold-start regimes: unseen drugs, unseen targets, and novel pairs where *both* entities are unseen.
2. **Benchmark naive baselines against ML models** (Random Forest and Deep Neural Networks) to demonstrate how much predictive signal stems purely from marginal label rates.
3. **Quantify domain-shift performance decay** as an explicit function of structural Tanimoto distance (ECFP4 fingerprints) and target sequence cosine distance (composition vectors).

---

## Dataset

This benchmark utilizes experimental binding affinities extracted from **BindingDB**:

- **Source:** [BindingDB](https://www.bindingdb.org/)
- **Version / Release:** <!-- TODO: Specify exact BindingDB release version (e.g., BindingDB 2024m3) -->
- **Download Date:** <!-- TODO: Specify dataset download date (YYYY-MM-DD) -->
- **Licence:** <!-- TODO: Specify BindingDB licence terms (e.g., Creative Commons Attribution 3.0 (CC BY 3.0)) -->
- **Citation:** <!-- TODO: Add formal BindingDB citation (e.g., Liu T. et al., Nucleic Acids Res., 2007) -->

### Curation Pipeline:
1. **Filtering (`filter_bindingdb.py`):**
   - Chunks through the 9 GB `BindingDB_All.tsv` source file.
   - Retains only human targets (`Homo sapiens`) with valid SMILES, complete amino acid sequences, and defined $K_i$ or $K_d$ measurements.
2. **Binary Affinity Thresholding (`curate_dti_binary.py`):**
   - Resolves canonical SMILES with RDKit and removes duplicate drug-target pairs.
   - **Strong Binders (Class 1):** Binding affinity $\le 100\text{ nM}$ ($\text{pAffinity} \ge 7.0$).
   - **Weak / Non-Binders (Class 0):** Binding affinity $\ge 10\ \mu\text{M}$ ($\text{pAffinity} \le 5.0$).
   - **Intermediate affinity values** ($100\text{ nM} < \text{affinity} < 10\ \mu\text{M}$) are discarded to create an unambiguous classification boundary.
   - **Dataset Size:** 289,282 curated pairs across 176,283 unique compounds and 3,253 unique human targets (3,262 with sequence fallback; 72.03% positive rate).

---

## Setup & Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.11)
- ~8 GB available RAM (for full 150k training scale and batch feature assembly)

### 1. Create Virtual Environment
```bash
# Create environment
python -m venv .venv

# Activate environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## How to Run

### Option A: End-to-End Execution via Master Runner (Recommended)
Execute all pipeline stages sequentially using [run_all.py](file:///c:/Users/sonyn/OneDrive/Documents/dti-project/run_all.py):
```bash
python run_all.py --train-size 150000
```
- By default, `run_all.py` **skips any step whose expected output already exists**, making subsequent runs near-instantaneous.
- **Note on Final Outputs:** Steps 1 (`filter_bindingdb.py`) and 2 (`curate_dti_binary.py`) are skipped even when `--force` is passed, as their curated output files (`bindingdb_human_ki_kd.tsv` and `bindingdb_human_dti_binary.tsv`) are final and the original 9 GB raw archive is not needed.
- To re-run downstream feature extraction, splitting, training, and analysis:
  ```bash
  python run_all.py --force --train-size 150000
  ```

### Option B: Step-by-Step Execution
```bash
# 1. Filter raw BindingDB (extracts human targets with Ki/Kd)
python filter_bindingdb.py

# 2. Curate binary pairs (<=100 nM vs >=10 uM)
python curate_dti_binary.py

# 3. Precompute ECFP4 drug fingerprints & AAC/DPC protein features
python features.py

# 4. Generate the 4 benchmark splits
python splits.py

# 5. Run test suite to verify split separation and feature integrity
pytest tests/

# 6. Train baselines and ML models across all 4 regimes
python train.py --train-size 150000

# 7. Execute diagnostic and memorisation analysis
python analyse.py --train-size 150000
```

---

## Interactive Streamlit Dashboard

An interactive dashboard ([app.py](file:///c:/Users/sonyn/OneDrive/Documents/dti-project/app.py)) is provided to explore benchmark results and query drug-target pairs with real-time reliability diagnostics.

### Launching the Dashboard
```bash
# Using the project environment:
.\.venv\Scripts\python.exe -m streamlit run app.py

# Or if the virtual environment is activated:
streamlit run app.py
```

### Dashboard Pages
1. **Predict:**
   - **Interactive Inference:** Enter any small molecule SMILES (or choose from representative examples) and search across 3,253 curated human targets.
   - **Feature Pipeline:** Generates 2,048-bit ECFP4 fingerprints with RDKit and concatenates them with 420-dim target composition vectors (AAC + DPC) to query the saved Split 1 Random Forest (`results/rf_split1.joblib`).
   - **Real-Time Regime Diagnosis & Reliability Warning:** Checks whether the target protein and compound were seen in training, and evaluates maximum Tanimoto similarity against a 30,000-drug training reference set. Dynamically reports the expected ROC-AUC:
     - **0.98 ROC-AUC (Green):** Familiar protein target and high chemical similarity ($\ge 0.80$). Trustworthy for candidate ranking.
     - **0.86 ROC-AUC (Amber):** Familiar target, but chemically distant scaffold ($< 0.40$ Tanimoto). High predictive uncertainty.
     - **0.75 ROC-AUC (Amber):** Unseen protein target (Split 3 regime). Explicit warning that protein features mislead and scores reflect compound bias.
     - **0.71 ROC-AUC (Red):** Cold-start regime (both drug and protein unseen; Split 4 regime). Critical warning that predictions are near chance level and must not be trusted for screening.
2. **Results:**
   - Visualizes the full four-regime table with parsed 95% bootstrap confidence intervals.
   - Displays benchmark comparison figures ([`results/four_regime_chart.png`](file:///c:/Users/sonyn/OneDrive/Documents/dti-project/results/four_regime_chart.png) and [`results/auc_vs_similarity.png`](file:///c:/Users/sonyn/OneDrive/Documents/dti-project/results/auc_vs_similarity.png)).
   - Surfaces memorisation ratios, Split 4 3-seed robustness metrics (0.690 / 0.726 / 0.722, mean 0.713), and the 20 most confident false negatives and false positives.
   - Highlights the central findings: the 0.984 $\to$ 0.713 cold-start collapse, and the ablation discovery where drug features alone outperform the full model on unseen targets (0.866 vs 0.745).
3. **Dataset:**
   - Summarizes 289,282 interaction pairs, 176,283 unique compounds, and 3,253 targets with a 72.0% positive rate.
   - Visualizes exploratory histograms and target label distributions.
   - Details the biophysical rationale behind curation thresholds: $K_i/K_d \le 100\text{ nM}$ vs $\ge 10\ \mu\text{M}$, and why $\text{IC}_{50}$ and $\text{EC}_{50}$ values were excluded.

---

## The Four Generalization Regimes

All splits maintain an 80/20 train/test partition (with a 10% stratified validation set carved from training for early stopping) using a fixed random seed (`42`):

1. **Split 1 — Random Pairs (Warm-Start):**
   - Standard random shuffle of interaction pairs.
   - Overlap: Both drugs and targets are observed during training.
   - Represents the classic "memorisation" regime.
2. **Split 2 — Unseen Drugs (Virtual Screening):**
   - Grouped by `canonical_smiles` using `GroupShuffleSplit`.
   - Overlap: Known targets, novel compounds.
   - Simulates screening novel chemical libraries against established therapeutic targets.
3. **Split 3 — Unseen Proteins (Target Discovery):**
   - Grouped by UniProt ID (and sequence for unmapped entries) so all isoforms stay on one side.
   - Overlap: Known chemical scaffolds, novel biological targets.
   - Simulates profiling known drugs against uncharacterized or orphan proteins.
4. **Split 4 — Unseen Drugs & Unseen Proteins (Cold-Start):**
   - 20% of drugs and 20% of proteins held out independently.
   - Test set strictly consists of pairs where *both* the drug and protein are held out.
   - Training set strictly consists of pairs where *neither* is held out.
   - Discards mixed pairs (42.24% of the dataset) to ensure zero entity leakage.
   - Simulates true de novo interaction prediction.

---

## Models Evaluated

Every model is trained and evaluated using identical settings across all four splits:

1. **Dummy Baseline:**
   - Always predicts the training set's empirical positive rate (constant baseline, ROC-AUC = 0.50).
2. **Protein-Only Baseline:**
   - Predicts each target's historical training positive rate; defaults to the global rate for unseen targets.
3. **Drug-Only Baseline:**
   - Predicts each compound's historical training positive rate; defaults to the global rate for unseen compounds.
4. **Nearest-Protein Baseline:**
   - For novel targets (and seen targets excluding self), finds the most similar training protein by cosine similarity of amino acid & dipeptide composition features and predicts that nearest neighbor's training positive rate.
5. **RF drug-features-only (Ablation):**
   - 200 estimators on the 2,048-bit ECFP4 fingerprint alone to isolate small molecule structural signal.
6. **RF protein-features-only (Ablation):**
   - 200 estimators on the 420-dim protein composition vector alone to isolate target sequence signal.
7. **Random Forest Classifier (Full):**
   - 200 estimators, `max_features='sqrt'`, `class_weight='balanced'`, multi-threaded (`n_jobs=-1`).
   - Trained on concatenated 2,048-bit ECFP4 drug fingerprints + 420-dim protein composition features.
8. **PyTorch Deep MLP:**
   - Architecture: $2468 \to 512 \to 128 \to 1$ with ReLU activations, $0.3$ dropout, weighted BCE loss, Adam optimizer ($\text{lr} = 10^{-3}$), and early stopping on validation ROC-AUC.

---

## Benchmark Results (150k Training Pairs)

Evaluated on the test set of each split using uniform 150,000-sample training sets (`--train-size 150000`, with earlier 40,000-sample baselines archived in `results_40k/`) and 1,000-sample bootstrap 95% confidence intervals:

| Model | Split 1 (Random) ROC-AUC | Split 2 (Unseen Drugs) ROC-AUC | Split 3 (Unseen Proteins) ROC-AUC | Split 4 (Cold-Start) ROC-AUC | Performance Lost (Split 4 vs 1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dummy** | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 100.0% |
| **Protein-Only** | 0.8477 | 0.8419 | 0.5000 | 0.5000 | 100.0% |
| **Drug-Only** | 0.7995 | 0.5000 | 0.7844 | 0.5000 | 100.0% |
| **Nearest-Protein** | 0.5259 | 0.5282 | 0.4792 | 0.5092 | 64.35% |
| **RF drug-features-only** | 0.9545 | 0.8811 | 0.8659 | 0.7406 | **47.05%** |
| **RF protein-features-only** | 0.8528 | 0.8469 | 0.6377 | 0.6183 | 66.48% |
| **Random Forest (Full)** | 0.9842 | 0.9824 | 0.7453 | 0.6903 | **60.70%** |
| **PyTorch MLP** | 0.9641 | 0.8991 | 0.8505 | 0.7512 | **45.87%** |

*See [DATA_CARD.md](DATA_CARD.md) for full PR-AUC metrics, confidence intervals, memorisation ratios, and detailed analysis.*

---

## Key Result Files & Directory Layout

```
dti-project/
├── bindingdb_human_ki_kd.tsv          # Filtered human pairs from raw BindingDB (596k records)
├── bindingdb_human_dti_binary.tsv     # Curated binary classification dataset (289k pairs)
├── features/                          # Git-ignored precomputed feature representations
│   ├── drug_features.npy              # 2,048-bit ECFP4 fingerprints (uint8)
│   ├── protein_features.npy           # 420-dim AAC (20) + DPC (400) composition (float32)
│   └── dti_features.npz               # Portable compressed feature bundle
├── splits/                            # Git-ignored row index splits
│   ├── random.npz                     # Split 1 indices (train, val, test)
│   ├── unseen_drugs.npz               # Split 2 indices
│   ├── unseen_proteins.npz            # Split 3 indices
│   └── unseen_both.npz                # Split 4 indices
├── results/                           # Active benchmark outputs & visualizations (150k sample size)
│   ├── four_regime_table.csv          # Point estimates & 1,000-sample bootstrap 95% CIs
│   ├── four_regime_chart.png          # High-resolution grouped bar chart (ROC & PR AUC)
│   ├── memorisation.csv               # Excess AUC retention ratio & % performance lost
│   ├── auc_vs_similarity.png          # Dual-panel ROC-AUC decay vs Tanimoto & Cosine similarity
│   ├── worst_predictions.csv          # 20 most confident false negatives on Split 4
│   ├── worst_false_positives.csv      # 20 most confident false positives on Split 4
│   ├── rf_split1.joblib               # Compressed Random Forest model artifact (Split 1)
│   └── preds/                         # Raw prediction arrays (.npz & .csv) per regime
├── results_40k/                       # Preserved 40k benchmark baseline results
├── tests/
│   └── test_splits.py                 # Pytest suite asserting entity disjointness & balance
├── train.py                           # Training & bootstrap evaluation script
├── analyse.py                         # Diagnostic and similarity evaluation script
├── run_all.py                         # Master pipeline runner
├── app.py                             # Interactive Streamlit dashboard
├── DATA_CARD.md                       # Comprehensive dataset documentation
└── README.md                          # Project documentation (this file)
```
