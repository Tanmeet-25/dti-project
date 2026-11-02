#!/usr/bin/env python3
"""
DTI Benchmark Dashboard (Streamlit)

Pages
  1. Predict    - 3D rotatable structure, molecular properties, binding
                  probability, and an honest reliability verdict based on which
                  evaluation regime the query actually falls into.
  2. Screen     - rank a library of compounds against one target, the way the
                  model would really be used, with the caveats attached.
  3. Results    - four-regime comparison, an interactive decision threshold,
                  memorisation and seed tables, and the error lists.
  4. Dataset    - composition, skew and the labelling decisions.

Run with:
    streamlit run app.py          (or: python -m streamlit run app.py)

Loads the artefacts produced by run_all.py. Nothing is retrained here.
"""

import ast
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
DATA_TSV = "bindingdb_human_dti_binary.tsv"
FEATURES_DIR = "features"
SPLITS_DIR = "splits"
RESULTS_DIR = "results"
RF_MODEL = os.path.join(RESULTS_DIR, "rf_split1.joblib")

# Measured Random Forest ROC-AUC from the 150k run, used for the reliability
# verdict. Source: results/four_regime_table.csv and the 3-seed split-4 run.
MEASURED = {
    "warm_similar": 0.98,
    "warm_moderate": 0.93,
    "warm_novel": 0.86,
    "cold_protein": 0.75,
    "cold_both": 0.71,
}

# Chosen to span the similarity range so a demo can show every reliability
# tier. Tanimoto figures are against a 30,000-drug sample of the training set.
EXAMPLES = {
    "Caffeine — chemistry the model knows (≈1.00)":
        "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Imatinib — kinase inhibitor (≈0.78)":
        "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "Aspirin — small and common (≈0.59)":
        "CC(=O)Oc1ccccc1C(=O)O",
    "Gefitinib — EGFR inhibitor (≈0.51)":
        "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "Cubane diol — scaffold unlike the training set (≈0.36)":
        "OCC12CC3(CO)CC(C1)CC(C2)C3",
    "Fluorinated chain — far outside the training chemistry (≈0.21)":
        "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)O",
}

st.set_page_config(page_title="DTI Benchmark", page_icon="🧬", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem;}
      [data-testid="stMetricValue"] {font-size: 1.9rem;}
      .verdict {border-radius: 10px; padding: 0.9rem 1.1rem; margin: 0.4rem 0 0.2rem 0;
                border-left: 6px solid; font-size: 0.95rem; line-height: 1.5;}
      .v-good {background: #10331f; border-color: #2ea043; color: #d6f5e0;}
      .v-mid  {background: #3a2f10; border-color: #d29922; color: #f8ebc8;}
      .v-bad  {background: #3d1d1d; border-color: #f85149; color: #ffd9d6;}
      .molbox {border: 1px solid rgba(250,250,250,0.15); border-radius: 10px;
               overflow: hidden; background: #0e1117;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset...")
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(
        DATA_TSV,
        sep="\t",
        usecols=[
            "canonical_smiles", "protein_sequence", "label",
            "target_name", "uniprot_id", "ligand_name",
        ],
        low_memory=False,
    )


@st.cache_resource(show_spinner="Loading model and features...")
def load_model_and_features():
    import joblib

    rf = joblib.load(RF_MODEL)
    drug_feats = np.load(os.path.join(FEATURES_DIR, "drug_features.npy"), mmap_mode="r")
    prot_feats = np.load(os.path.join(FEATURES_DIR, "protein_features.npy"), mmap_mode="r")
    with open(os.path.join(FEATURES_DIR, "drug_to_idx.json"), encoding="utf-8") as f:
        drug_to_idx = json.load(f)
    with open(os.path.join(FEATURES_DIR, "protein_to_idx.json"), encoding="utf-8") as f:
        prot_to_idx = json.load(f)
    return rf, drug_feats, prot_feats, drug_to_idx, prot_to_idx


@st.cache_resource(show_spinner="Indexing split-1 training data...")
def load_training_reference(_df, _drug_feats, _drug_to_idx, sample_size=30000):
    """
    What the saved model actually saw in training (split 1), so the app can tell
    the user which evaluation regime a query falls into. The fingerprint matrix
    is a random sample to keep memory low, so max-similarity is a slight
    under-estimate; the UI says so.
    """
    train_idx = np.load(os.path.join(SPLITS_DIR, "random.npz"))["train_idx"]
    train_smiles = set(_df["canonical_smiles"].iloc[train_idx].unique())
    train_seqs = set(_df["protein_sequence"].iloc[train_idx].unique())

    rng = np.random.RandomState(42)
    smiles_list = list(train_smiles)
    if len(smiles_list) > sample_size:
        pick = rng.choice(len(smiles_list), size=sample_size, replace=False)
        smiles_list = [smiles_list[i] for i in pick]
    rows = [_drug_to_idx[s] for s in smiles_list if s in _drug_to_idx]
    fp_sample = np.asarray(_drug_feats[rows], dtype=np.float32)
    return train_smiles, train_seqs, fp_sample


@st.cache_data(show_spinner=False)
def load_results_table() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS_DIR, "four_regime_table.csv"))


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS_DIR, name))


@st.cache_data(show_spinner=False)
def load_predictions(split: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS_DIR, "preds", f"{split}_predictions.csv"))


@st.cache_data(show_spinner="Building target list...")
def build_target_options(_df: pd.DataFrame) -> pd.DataFrame:
    g = (
        _df.groupby("protein_sequence")
        .agg(
            target_name=("target_name", "first"),
            uniprot_id=("uniprot_id", "first"),
            pairs=("label", "size"),
            pos_rate=("label", "mean"),
        )
        .reset_index()
        .sort_values("pairs", ascending=False)
    )
    g["display"] = (
        g["target_name"].fillna("Unnamed target").astype(str).str.slice(0, 68)
        + "  ·  " + g["uniprot_id"].fillna("no UniProt").astype(str)
        + "  ·  " + g["pairs"].astype(str) + " pairs"
    )
    return g


# --------------------------------------------------------------------------
# Chemistry helpers
# --------------------------------------------------------------------------
def ecfp4(smiles: str):
    """Canonicalise a SMILES and return (canonical, 2048-bit ECFP4, mol)."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return Chem.MolToSmiles(mol), gen.GetFingerprintAsNumPy(mol).astype(np.uint8), mol


@st.cache_data(show_spinner=False)
def molblock_3d(smiles: str):
    """Generate 3D coordinates. Returns a MOL block, or None if embedding fails."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=400)
    except Exception:
        pass
    return Chem.MolToMolBlock(mol)


@st.cache_data(show_spinner=False)
def svg_2d(smiles: str, width=380, height=260) -> str:
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.setBackgroundColour((0.055, 0.067, 0.09, 0))
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def show_3d(molblock: str, style="stick", spin=True, height=380):
    """Embed an interactive, rotatable 3D viewer."""
    import py3Dmol

    view = py3Dmol.view(width="100%", height=height)
    view.addModel(molblock, "mol")
    styles = {
        "stick": {"stick": {"radius": 0.14}},
        "ball and stick": {"stick": {"radius": 0.1}, "sphere": {"scale": 0.24}},
        "space filling": {"sphere": {"scale": 0.95}},
        "wireframe": {"line": {"linewidth": 2.5}},
    }
    view.setStyle(styles.get(style, styles["stick"]))
    view.setBackgroundColor("#0e1117")
    if spin:
        view.spin(True)
    view.zoomTo()
    components.html(view._make_html(), height=height + 10)


@st.cache_data(show_spinner=False)
def properties(smiles: str) -> dict:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    breaches = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return {
        "Molecular weight": f"{mw:.1f}",
        "logP": f"{logp:.2f}",
        "TPSA (Å²)": f"{rdMolDescriptors.CalcTPSA(mol):.1f}",
        "H-bond donors": hbd,
        "H-bond acceptors": hba,
        "Rotatable bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "Heavy atoms": mol.GetNumHeavyAtoms(),
        "Rings": rdMolDescriptors.CalcNumRings(mol),
        "Lipinski breaches": breaches,
    }


def max_tanimoto(query_fp: np.ndarray, fp_matrix: np.ndarray) -> float:
    q = query_fp.astype(np.float32)
    inter = fp_matrix @ q
    union = fp_matrix.sum(axis=1) + q.sum() - inter
    return float((inter / np.maximum(union, 1.0)).max())


def regime_verdict(protein_seen: bool, drug_seen: bool, similarity: float):
    """Map a query onto the evaluation regime it belongs to."""
    if not protein_seen and not drug_seen:
        return ("Both the protein and the drug are new to the model",
                MEASURED["cold_both"], "v-bad",
                "Treat the score as a weak hint, nothing more.")
    if not protein_seen:
        return ("This protein is new to the model",
                MEASURED["cold_protein"], "v-bad",
                "Useful only for coarse ranking.")
    if drug_seen or similarity >= 0.8:
        return ("Familiar protein, familiar chemistry",
                MEASURED["warm_similar"], "v-good",
                "This is the model's strongest setting.")
    if similarity < 0.4:
        return ("Familiar protein, but chemistry unlike anything in training",
                MEASURED["warm_novel"], "v-mid",
                "Outside the applicability domain; expect real errors.")
    return ("Familiar protein, moderately familiar chemistry",
            MEASURED["warm_moderate"], "v-mid",
            "Reasonable for ranking candidates.")


def parse_ci(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return (np.nan, np.nan)
    return value


def pretty_ligand_name(raw, fallback: str) -> str:
    if not isinstance(raw, str) or not raw.strip() or raw == "nan":
        return fallback[:28]
    parts = [p.strip() for p in raw.split("::") if p.strip()]
    short = [p for p in parts if len(p) <= 32 and not p.startswith("Scaffold")]
    return (short[-1] if short else parts[0][:32]) if parts else fallback[:28]


# --------------------------------------------------------------------------
# Page 1 - Predict
# --------------------------------------------------------------------------
def page_predict():
    st.title("Predict a drug–target interaction")
    st.caption(
        "Classifies strong binders (Ki/Kd ≤ 100 nM) against weak ones (≥ 10 µM). "
        "It ranks candidates for laboratory testing; it does not establish that "
        "any compound binds any target."
    )

    df = load_dataset()
    rf, drug_feats, prot_feats, drug_to_idx, prot_to_idx = load_model_and_features()
    train_smiles, train_seqs, fp_sample = load_training_reference(df, drug_feats, drug_to_idx)
    targets = build_target_options(df)

    left, right = st.columns(2)
    with left:
        st.subheader("1. Drug")
        example = st.selectbox("Load an example", ["(type my own)"] + list(EXAMPLES))
        smiles = st.text_area("SMILES", value=EXAMPLES.get(example, ""), height=88)
    with right:
        st.subheader("2. Protein target")
        choice = st.selectbox("Search by name or UniProt ID", targets["display"].tolist())
        trow = targets[targets["display"] == choice].iloc[0]
        st.caption(
            f"{trow['pairs']:,} measured pairs · {trow['pos_rate'] * 100:.0f}% of them "
            f"are strong binders · sequence {len(trow['protein_sequence']):,} residues"
        )

    go = st.button("Predict", type="primary", use_container_width=True)
    if not go:
        st.info("Pick an example and a target, then press Predict.")
        return
    if not smiles.strip():
        st.warning("Enter a SMILES string first.")
        return

    canonical, fp, _ = ecfp4(smiles.strip())
    if fp is None:
        st.error("RDKit could not parse that SMILES. Check the structure and try again.")
        return

    sequence = trow["protein_sequence"]
    x = np.concatenate([
        fp.astype(np.float32),
        np.asarray(prot_feats[prot_to_idx[sequence]], dtype=np.float32),
    ]).reshape(1, -1)
    probability = float(rf.predict_proba(x)[0, 1])

    drug_seen = canonical in train_smiles
    protein_seen = sequence in train_seqs
    similarity = max_tanimoto(fp, fp_sample)
    headline, expected, css, advice = regime_verdict(protein_seen, drug_seen, similarity)

    st.divider()
    viz, info = st.columns([1.05, 1])

    with viz:
        tab3d, tab2d = st.tabs(["3D structure", "2D structure"])
        with tab3d:
            c1, c2 = st.columns([2, 1])
            style = c1.selectbox(
                "Style", ["ball and stick", "stick", "space filling", "wireframe"],
                label_visibility="collapsed",
            )
            spin = c2.toggle("Spin", value=True)
            block = molblock_3d(canonical)
            if block:
                st.markdown('<div class="molbox">', unsafe_allow_html=True)
                show_3d(block, style=style, spin=spin)
                st.markdown("</div>", unsafe_allow_html=True)
                st.caption("Drag to rotate · scroll to zoom · right-drag to pan")
            else:
                st.warning("Could not generate 3D coordinates for this molecule.")
        with tab2d:
            svg = svg_2d(canonical)
            if svg:
                components.html(
                    f'<div style="display:flex;justify-content:center;'
                    f'background:#0e1117;border-radius:10px;padding:6px">{svg}</div>',
                    height=285,
                )

    with info:
        a, b, c = st.columns(3)
        a.metric("Binding probability", f"{probability:.3f}")
        b.metric("Closest training drug", f"{similarity:.2f}")
        c.metric("Expected ROC-AUC here", f"{expected:.2f}")

        st.markdown(
            f'<div class="verdict {css}"><b>{headline}.</b> On our held-out tests the '
            f'model scored about <b>{expected:.2f} ROC-AUC</b> on queries like this. '
            f'{advice}</div>',
            unsafe_allow_html=True,
        )
        st.progress(min(probability, 1.0), text=f"Strong-binder probability {probability:.1%}")

        props = properties(canonical)
        if props:
            st.markdown("**Molecular properties**")
            items = list(props.items())
            for row_start in range(0, len(items), 3):
                cols = st.columns(3)
                for col, (k, v) in zip(cols, items[row_start:row_start + 3]):
                    col.metric(k, v)

    with st.expander("Why these numbers, and what they do not mean"):
        st.markdown(
            f"""
- **Canonical SMILES:** `{canonical}`
- **Drug seen in training:** {"yes" if drug_seen else "no"} ·
  **Protein seen in training:** {"yes" if protein_seen else "no"}
- Similarity is the highest Tanimoto score against a 30,000-drug random sample
  of the training set, so the true closest match may be slightly higher.
- The saved model was trained on the **random split**, the optimistic setting.
  Its 0.98 ROC-AUC does not carry over: the same model scores about 0.75 on new
  proteins and about 0.71 when the drug is new too.
- Probabilities sit in a narrow band because the forest uses
  `class_weight="balanced"`. Compare them with each other, not against 0.5.
- The 3D conformer is one low-energy shape from a force field. It is for looking
  at — the model never sees 3D structure, only the 2D fingerprint.
            """
        )


# --------------------------------------------------------------------------
# Page 2 - Virtual screening
# --------------------------------------------------------------------------
def page_screen():
    st.title("Virtual screening")
    st.caption(
        "Rank a library of compounds against one target — the task this kind of "
        "model is actually built for."
    )

    df = load_dataset()
    rf, drug_feats, prot_feats, drug_to_idx, prot_to_idx = load_model_and_features()
    train_smiles, train_seqs, fp_sample = load_training_reference(df, drug_feats, drug_to_idx)
    targets = build_target_options(df)

    c1, c2 = st.columns([2, 1])
    choice = c1.selectbox("Target", targets["display"].tolist())
    trow = targets[targets["display"] == choice].iloc[0]
    library_size = c2.slider("Compounds to screen", 200, 5000, 1000, step=200)

    only_untested = st.checkbox(
        "Only compounds never measured against this target", value=True,
        help="Otherwise known pairs can appear, and the model has seen their answer.",
    )

    if not st.button("Run screen", type="primary", use_container_width=True):
        st.info("Choose a target and press Run screen.")
        return

    sequence = trow["protein_sequence"]
    rng = np.random.RandomState(42)
    pool = df["canonical_smiles"].unique()
    if only_untested:
        tested = set(df.loc[df["protein_sequence"] == sequence, "canonical_smiles"])
        pool = np.array([s for s in pool if s not in tested], dtype=object)
    if len(pool) > library_size:
        pool = rng.choice(pool, size=library_size, replace=False)

    with st.spinner(f"Scoring {len(pool):,} compounds..."):
        rows = [drug_to_idx[s] for s in pool]
        prot_vec = np.asarray(prot_feats[prot_to_idx[sequence]], dtype=np.float32)
        X = np.empty((len(rows), 2468), dtype=np.float32)
        X[:, :2048] = np.asarray(drug_feats[rows], dtype=np.float32)
        X[:, 2048:] = prot_vec
        scores = rf.predict_proba(X)[:, 1]

    protein_seen = sequence in train_seqs
    out = pd.DataFrame({"canonical_smiles": pool, "probability": scores})
    names = df.drop_duplicates("canonical_smiles").set_index("canonical_smiles")["ligand_name"]
    out["compound"] = [pretty_ligand_name(names.get(s), s) for s in out["canonical_smiles"]]
    out = out.sort_values("probability", ascending=False).reset_index(drop=True)

    top = out.head(15).copy()
    top["similarity to training"] = [
        round(max_tanimoto(np.asarray(drug_feats[drug_to_idx[s]], dtype=np.float32), fp_sample), 2)
        for s in top["canonical_smiles"]
    ]
    top.insert(0, "rank", np.arange(1, len(top) + 1))

    a, b, c = st.columns(3)
    a.metric("Compounds screened", f"{len(out):,}")
    b.metric("Scored above 0.50", f"{(out['probability'] > 0.5).sum():,}")
    c.metric("Target seen in training", "yes" if protein_seen else "no")

    if not protein_seen:
        st.markdown(
            '<div class="verdict v-bad"><b>This target was not in training.</b> '
            'On unseen proteins the model scores about 0.75 ROC-AUC, so this ranking '
            'is weak. Treat it as a rough ordering, not a shortlist.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="verdict v-good"><b>This target was in training.</b> '
            'Ranking novel compounds against a familiar target is the model\'s best '
            'case (about 0.98 ROC-AUC on the unseen-drugs split).</div>',
            unsafe_allow_html=True,
        )

    st.subheader("Top candidates")
    st.dataframe(
        top[["rank", "compound", "probability", "similarity to training", "canonical_smiles"]],
        use_container_width=True, hide_index=True,
        column_config={
            "probability": st.column_config.ProgressColumn(
                "probability", min_value=0.0, max_value=1.0, format="%.3f"),
        },
    )

    st.subheader("Inspect a hit")
    labels = [
        f"#{r.rank}  {r.compound}  —  {r.probability:.3f}"
        for r in top.itertuples()
    ]
    position = st.selectbox(
        "Compound", range(len(labels)), format_func=lambda i: labels[i]
    )
    hit = top.iloc[position]
    v, d = st.columns([1, 1])
    with v:
        block = molblock_3d(hit["canonical_smiles"])
        if block:
            st.markdown('<div class="molbox">', unsafe_allow_html=True)
            show_3d(block, style="ball and stick", spin=True, height=320)
            st.markdown("</div>", unsafe_allow_html=True)
    with d:
        st.metric("Probability", f"{hit['probability']:.3f}")
        st.metric("Closest training drug", f"{hit['similarity to training']:.2f}")
        st.code(hit["canonical_smiles"], language=None)

    st.caption(
        "Scores are hypotheses for experimental follow-up. A high score means the "
        "compound resembles things that bound similar targets before — not that it binds."
    )


# --------------------------------------------------------------------------
# Page 3 - Results
# --------------------------------------------------------------------------
def page_results():
    st.title("Four evaluation regimes")
    st.markdown("Identical models and settings throughout. **The split is the only variable.**")

    table = load_results_table()
    rf_rows = table[table["Model"] == "Random Forest"].set_index("Split")["ROC-AUC"]

    a, b, c, d = st.columns(4)
    a.metric("Random split", f"{rf_rows.get('random', np.nan):.3f}")
    b.metric("Unseen drugs", f"{rf_rows.get('unseen_drugs', np.nan):.3f}")
    c.metric("Unseen proteins", f"{rf_rows.get('unseen_proteins', np.nan):.3f}")
    d.metric("Unseen both", "0.713", delta="−0.271 vs random", delta_color="inverse")
    st.caption(
        "Random Forest, ROC-AUC. The cold-start figure is the mean of three seeds "
        "(0.690 / 0.726 / 0.722); every other figure is a single seed."
    )

    st.info(
        "**Headline:** the Random Forest loses about **56%** of its above-chance "
        "performance between the random split and the cold-start split. Quadrupling "
        "the training data raised the random-split score from 0.962 to 0.984 and left "
        "cold-start unchanged."
    )

    pretty = {
        "random": "1. Random pairs", "unseen_drugs": "2. Unseen drugs",
        "unseen_proteins": "3. Unseen proteins", "unseen_both": "4. Unseen drugs + proteins",
    }

    st.subheader("Where would you set the cut-off?")
    st.caption(
        "Everything above reports ranking quality. In a real screen you must pick a "
        "threshold — drag it and watch precision trade against recall on the real "
        "test predictions."
    )
    tc1, tc2 = st.columns([1, 1])
    split_pick = tc1.selectbox("Split", list(pretty), format_func=lambda s: pretty[s])
    threshold = tc2.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)

    try:
        preds = load_predictions(split_pick)
        y = preds["y_true"].values
        p = preds["random_forest"].values
        flagged = p >= threshold
        tp = int((flagged & (y == 1)).sum())
        fp = int((flagged & (y == 0)).sum())
        fn = int((~flagged & (y == 1)).sum())
        tn = int((~flagged & (y == 0)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", f"{precision:.3f}",
                  help="Of the pairs flagged as binders, how many really are.")
        m2.metric("Recall", f"{recall:.3f}",
                  help="Of the real binders, how many were flagged.")
        m3.metric("F1", f"{f1:.3f}")
        m4.metric("Flagged", f"{int(flagged.sum()):,} of {len(y):,}")
        st.dataframe(
            pd.DataFrame(
                [[tp, fn], [fp, tn]],
                index=["Actually binds", "Actually weak"],
                columns=["Predicted binder", "Predicted weak"],
            ),
            use_container_width=True,
        )
        st.caption(
            f"At threshold {threshold:.2f} on the {pretty[split_pick].lower()} split, "
            f"{fp:,} weak compounds would be sent for testing and {fn:,} real binders missed."
        )
    except FileNotFoundError:
        st.caption("Prediction files not found — run train.py first.")

    st.subheader("Full comparison")
    view = table.copy()
    view["ROC-AUC (95% CI)"] = [
        f"{v:.3f} [{parse_ci(ci)[0]:.3f}, {parse_ci(ci)[1]:.3f}]"
        for v, ci in zip(view["ROC-AUC"], view["ROC-AUC 95% CI"])
    ]
    view["PR-AUC (95% CI)"] = [
        f"{v:.3f} [{parse_ci(ci)[0]:.3f}, {parse_ci(ci)[1]:.3f}]"
        for v, ci in zip(view["PR-AUC"], view["PR-AUC 95% CI"])
    ]
    view["Regime"] = view["Split"].map(pretty).fillna(view["Split"])
    st.dataframe(
        view[["Regime", "Model", "ROC-AUC (95% CI)", "PR-AUC (95% CI)", "Train Time (s)"]],
        use_container_width=True, hide_index=True,
    )

    for caption, filename in [
        ("Every model across every regime", "four_regime_chart.png"),
        ("Accuracy falls as test entities get less similar to training ones",
         "auc_vs_similarity.png"),
    ]:
        path = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(path):
            st.subheader(caption)
            st.image(path, use_container_width=True)

    st.subheader("What the model is really using")
    st.markdown(
        """
On **new proteins**, a Random Forest given *only* the drug fingerprint scores
**0.866** — higher than the full model's **0.745**. Protein composition features
help only while the protein is familiar; once it is new, they hurt.

That is the central result: the apparent skill is largely **compound-level bias**
(published compounds tend to be potent ones), not a learned rule about how a drug
and a protein interact.
        """
    )

    cols = st.columns(2)
    for col, (title, filename) in zip(
        cols, [("Memorisation by model", "memorisation.csv"),
               ("Seed robustness, split 4", "split4_rf_seed_robustness.csv")]):
        if os.path.exists(os.path.join(RESULTS_DIR, filename)):
            with col:
                st.markdown(f"**{title}**")
                st.dataframe(load_csv(filename), use_container_width=True, hide_index=True)

    st.subheader("Where it fails")
    st.caption(
        "On the cold-start split the model is systematically cautious: true binders "
        "are pushed as low as 0.155, while no non-binder ever rises above 0.615."
    )
    t1, t2 = st.tabs(["Missed binders", "False alarms"])
    for tab, filename in [(t1, "worst_predictions.csv"), (t2, "worst_false_positives.csv")]:
        with tab:
            if os.path.exists(os.path.join(RESULTS_DIR, filename)):
                st.dataframe(load_csv(filename), use_container_width=True, hide_index=True)
            else:
                st.caption("File not found.")


# --------------------------------------------------------------------------
# Page 4 - Dataset
# --------------------------------------------------------------------------
def page_dataset():
    st.title("The dataset")
    st.caption("Human targets from BindingDB with a measured Ki or Kd value.")

    df = load_dataset()
    n = len(df)
    pos = int((df["label"] == 1).sum())
    drugs = df["canonical_smiles"].nunique()
    prots = df["uniprot_id"].dropna().nunique()

    a, b, c, d = st.columns(4)
    a.metric("Interaction pairs", f"{n:,}")
    b.metric("Unique drugs", f"{drugs:,}")
    c.metric("Unique proteins", f"{prots:,}")
    d.metric("Strong binders", f"{pos / n * 100:.1f}%")

    st.warning(
        f"**The data is lopsided, and that shapes every result.** {drugs:,} drugs but "
        f"only {prots:,} proteins, so most drugs appear once or twice while each protein "
        f"appears in many pairs. Of the 1,053 proteins with at least 20 pairs, 423 (40%) "
        f"are almost entirely strong binders or almost entirely weak ones — so a model can "
        f"score well by recognising the protein alone, without looking at the drug at all."
    )

    for caption, filename in [
        ("Most-tested proteins and compounds", "most_frequent_drugs_and_proteins.png"),
        ("How many pairs each drug and protein has (log scale)",
         os.path.join(RESULTS_DIR, "pairs_per_drug_and_protein_histograms.png")),
        ("Proteins that are nearly all strong or nearly all weak binders",
         os.path.join(RESULTS_DIR, "protein_fraction_labelled_1_histogram.png")),
    ]:
        if os.path.exists(filename):
            st.subheader(caption)
            st.image(filename, use_container_width=True)

    st.subheader("How the labels were made")
    st.markdown(
        """
| Decision | Choice | Reason |
| --- | --- | --- |
| Scope | Human targets, Ki and Kd only | Ki/Kd measure binding directly; IC50 depends on the assay |
| Strong binder (1) | ≤ 100 nM | A clear two-order-of-magnitude gap from the negatives |
| Weak binder (0) | ≥ 10 µM | Measured weak binders, not random untested pairs |
| In between | Dropped | Borderline values would add label noise |
| `>500`-type values | Dropped | The true value could fall in the grey zone |
| Conflicting pairs | Dropped (1,269) | Experiments disagreed, so the label is unknown |
| Duplicates | One row per pair | Duplicates across train and test would leak |
        """
    )
    st.caption(
        "Verified on the final file: zero duplicate pairs, and no row whose label "
        "contradicts its affinity value."
    )


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
def main():
    missing = [p for p in (DATA_TSV, RF_MODEL, FEATURES_DIR, SPLITS_DIR, RESULTS_DIR)
               if not os.path.exists(p)]
    if missing:
        st.error(
            "Run this from the project folder — these are missing: " + ", ".join(missing)
            + ".\n\nGenerate them with `python run_all.py --train-size 150000`."
        )
        st.stop()

    with st.sidebar:
        st.title("🧬 DTI Benchmark")
        st.caption("Memorisation vs generalisation in drug–target prediction")
        page = st.radio(
            "Page", ["Predict", "Screen", "Results", "Dataset"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(
            "Anuj Seniwal · Jashmann Singh · Vardaan Makwana · Yash Sheth\n\n"
            "Predictions are hypotheses for laboratory testing, not evidence of binding."
        )

    {"Predict": page_predict, "Screen": page_screen,
     "Results": page_results, "Dataset": page_dataset}[page]()


if __name__ == "__main__":
    main()