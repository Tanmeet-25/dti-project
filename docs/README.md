# DTI Benchmark — showcase site

One self-contained page. No build step, no dependencies to install.

## Deploy to Vercel

**Easiest:** go to vercel.com/new, drag this whole folder onto the page, press Deploy.
You get a URL like `dti-benchmark.vercel.app` in under a minute.

**From GitHub:** push this folder to a repo, then on vercel.com/new choose
"Import Git Repository", pick it, and press Deploy. Leave every build setting empty —
Vercel serves `index.html` as a static site. Pushes to `main` redeploy automatically.

**From the terminal:**
```
npm i -g vercel
cd deploy
vercel
```

## Editing

Everything lives in `index.html`: styles in the `<style>` block at the top, content in
the markup, data and charts in the `<script>` at the bottom. The numbers the charts
draw are in the `REGIMES`, `MODELS`, `SIM1` and `SIM2` objects — change a value there
and the chart follows.

The molecules in `MOLS` carry pre-computed 3D coordinates. To add one, generate a
MOL block with RDKit:

```python
from rdkit import Chem
from rdkit.Chem import AllChem
m = Chem.AddHs(Chem.MolFromSmiles("YOUR_SMILES"))
p = AllChem.ETKDGv3(); p.randomSeed = 42
AllChem.EmbedMolecule(m, p); AllChem.MMFFOptimizeMolecule(m)
print(Chem.MolToMolBlock(m))
```

## External resources

Google Fonts (Archivo, Spectral, IBM Plex Mono) and 3Dmol.js load from CDNs. If the
3D viewer cannot load, each molecule falls back to an embedded 2D drawing.
