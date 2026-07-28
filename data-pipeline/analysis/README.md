# analysis/ — not part of the *pipeline* build, but part of the paper

Nothing in here is run by `make all` or `scripts/run_all.ps1`, and nothing in
`src/` imports it. But do not read that as "optional": since `models/gnn_map.py`
was deleted as a duplicate, **`maps.py` is the script that produces the
manuscript's figures**, and `moran.py` produces its reported spatial-autocorrelation
statistic. Both are stage 3 of the workflow in the root README.

| file | what it is |
|---|---|
| `maps.py` | **the** GNN figure generator — reads a checkpoint from `models/gnn.py` |
| `moran.py` | Global Moran's I + an OLS table with spatial diagnostics; writes `ols_summary.txt` and `flow_map.png` |
| `ols_summary.txt` | output of a past `moran.py` run |
| `flow_map.png` | output of a past `moran.py` run |

Because these two are now paper-critical while living in a directory the pipeline
ignores, consider moving them to `models/` so the layout matches the workflow.

## Inputs

Both scripts read the modelling splits, which are **not** produced by this
pipeline — they come from `models/make_splits.py`, which consumes this
pipeline's `data/processed/mx_dyadic_positive.csv`. Build them first:

```bash
cd data-pipeline && make all      # produces mx_dyadic_positive.csv
cd .. && python models/make_splits.py   # produces data/X_train.csv etc.
```

Both scripts now resolve their paths from the repository root automatically
(`ROOT = Path(__file__).resolve().parent.parent.parent`), so they can be run from
anywhere, and both accept `--data-dir` / `--outdir` overrides.

## `maps.py` was one of two forks; it is now the only one

`maps.py` and `models/gnn_map.py` were forks of the same figure-generation script —
same model definition, same metrics, same figure names, written into the same
directory, so whichever ran last silently overwrote the other. `maps.py` was the
newer fork: it adds `binned_line_plot()` (quantile-binned means with
sample-size-scaled line width) and an `--income` flag, and its `climate_keywords`
list had already been fixed to match bare `temp`/`precip` names.

`models/gnn_map.py` has been deleted. Its pre-port version is still in git history
if it is ever needed.

No model figures are committed anywhere in the repository. A set of eight used to
live at `models/gnn_figures/`, but they were output from the model's earlier
**Brazilian** application — the maps drew Brazil, the gate plots were labelled in
BRL — so they have been deleted rather than left to be mistaken for results. Run
`maps.py` to regenerate; output goes to `models/runs/figs_<name>/`.

## The residual Moran's I was removed from the figure script

`maps.py` used to print a residual Moran's I computed by `moran_on_subgraph()`,
which returned impossible values above 1 (it reused full-graph row-standardised
weights on a 452-node subgraph, and took *N*, the mean and the variance over all
2,457 nodes). Both that function and its call site are gone.

The reported statistic comes from `moran.py` instead, via `esda.Moran`:
**Global Moran's I = 0.3051** (p=0.001, n=1,959) on out-migration flows by source
municipality, and 0.3199 on the OLS error. `moran.py` prints it to stdout as well
as drawing it on `flow_map.png`.

The gate-activation Moran statistics in `maps.py` are retained — those are computed
over the whole graph, where the weight normalisation is consistent.

## A correction

An earlier version of this file said `moran.py` was a stale fork superseded by a
copy at the repo root, `data/moran.py`, and to "prefer that one". **That was
backwards.** The copy here is the Mexico-ported one — it joins on `cvegeo`, reads
`data/processed/municipios_harmonized_2020_shp/`, and zero-pads to 5 characters.
The root `data/moran.py` was the older **Brazil** version (it imported `geobr`
and merged on `codigo_ibge` against a Brazilian municipality gazetteer); it has
been deleted along with the rest of the Brazil-era `data/` scripts.

The one input here that this pipeline does produce is
`data/processed/municipios_centroids.csv` (written by `03_distance.py`), which
`moran.py` reads for coordinates and which `make_splits.py` copies to
`data/muni_centroids.csv` for the models.
