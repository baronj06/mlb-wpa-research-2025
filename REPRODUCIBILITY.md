# Installation and reproducibility

Research phase is closed. Commands below reproduce or verify frozen results, not permission to collect data, retrain models or optimize candidates. Run from the repository root. Use an isolated copy for any command that regenerates historical analysis outputs; the finalization process preserves all prior results.

## 1. Python environment

Tested research environment: Python3.11.5, macOS arm64. Direct dependencies are in pyproject.toml; the original tested snapshot is requirements-lock.txt.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e '.[dev,notebooks]'
python -m pip check
```

The lock describes one platform, so other Python/OS combinations may need compatible wheels. No dependency update was needed for final research tests. Optional PDF packaging dependencies are separately pinned in requirements-portfolio.txt; they do not change the original research lock.

```bash
python -m pip install -r requirements-portfolio.txt
```

The desktop finalization used the existing bundled PDF runtime, not a change to the project's research environment. Record its versions in verification/runtime.json. For portable use, install the optional pins in your own environment. Matplotlib's bundled DejaVu fonts are embedded by the PDF renderer.

## 2. Read and verify the final portfolio (no raw cache)

```bash
.venv/bin/python scripts/verify_final_portfolio.py
```

This verifies artifact/source mappings, hashes, reported values, CSV identities and missingness, without downloads or fitting. PDFs are checked using pypdf when available. Files copied from historical outputs are never recomputed by the assembly script.

## 3. Rebuild the final presentation only

Requires existing tracked result tables/charts and study evidence files; does not require raw play responses.

```bash
.venv/bin/python scripts/build_final_portfolio.py
.venv/bin/python scripts/render_final_portfolio.py
.venv/bin/python scripts/seal_final_portfolio.py
.venv/bin/python scripts/verify_final_portfolio.py
```

The builder assembles Markdown and byte-identical asset copies. The renderer uses reportlab/pypdf and embeds fonts. The sealer records hashes after a successful test run. Font/PDF runtime differences can change PDF bytes; scientific CSV/PNG source identities must remain exact. Rendered QA images are temporary and ignored.

## 4. Fixed-data analytical reproduction (isolated copy)

Requires the compact frozen season Parquet, official player totals, cached per-game batting JSON, saved study designs and model/reference summaries. No new WE/LI fit occurs in the commands below, but they regenerate study outputs; do not run them in a preservation-only archive.

```bash
MPLCONFIGDIR=.cache/matplotlib OPENBLAS_NUM_THREADS=1 .venv/bin/python -m mlb_wpa.eows_research
MPLCONFIGDIR=.cache/matplotlib OPENBLAS_NUM_THREADS=1 .venv/bin/python -m mlb_wpa.eows_interpretation
MPLCONFIGDIR=.cache/matplotlib OPENBLAS_NUM_THREADS=1 .venv/bin/python -m mlb_wpa.eows_v11_study
```

The original framework/splits/seeds remain fixed. Do not rerun design generation to select another split after observing evaluation. The v1.1 study uses1565 development and865 isolated-role evaluation games; D remains gated by development evidence.

## 5. Full integration tests and event reconstruction (complete cache)

```bash
.venv/bin/python scripts/run_final_tests.py
.venv/bin/python scripts/verify_final_portfolio.py
```

The test wrapper restores legacy fixture-generated reference reports, writes new test outputs only under final_portfolio/verification and checks all protected prior research hashes. Full integration tests require the preserved source cache and historical fixtures. Missing inputs should be reported, never replaced by new downloads or fabricated values.

The frozen model application command mlb_wpa.eows_apply is documented in earlier work but is not required to rebuild final artifacts and was not run during finalization. Source-to-event reconstruction needs original cached raw responses and original receipts. FanGraphs automatic requests remain disabled. The final verification does not claim cache-free source-level reproducibility.
