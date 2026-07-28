# analysis/ — not part of the build

Nothing in here is run by `make all` or `scripts/run_all.ps1`, and nothing in
`src/` imports it. These are downstream analysis scripts that were sitting loose
in `data/` (and `ols_summary.txt` at the pipeline root); they were moved here so
`data/` holds only data.

| file | what it is |
|---|---|
| `moran.py` | Moran's I on the flow residuals + an OLS table; writes `ols_summary.txt` and `flow_map.png` |
| `maps.py` | GNN flow map / choropleth plotting |
| `ols_summary.txt` | output of a past `moran.py` run |
| `flow_map.png` | output of a past `moran.py` run |

**Their paths are relative to the repository root, not to `data-pipeline/`.**
They read things like `data/X_train.csv`, `data/processed/X_train_mx.csv` and
`data/muni_centroids.csv` — the modelling splits produced outside this pipeline —
so running them from `data-pipeline/` will not find their inputs. Run them from
the repo root, or fix the paths first.

`moran.py` is also a **stale fork**: a newer copy lives at the repo root as
`data/moran.py`. Prefer that one; this copy is kept only so nothing is lost.

The one file here this pipeline does produce is
`data/processed/municipios_centroids.csv` (written by `03_distance.py`), which
`moran.py` reads for coordinates.
