# Spatially-Gated Deep GNN — internal migration in Mexico

Replication package for a study of municipality-to-municipality internal
migration in Mexico (2015→2020), combining:

- **`data-pipeline/`** — a reproducible pipeline that builds a dyadic
  origin–destination panel over 2,457 harmonized municipalities (6,034,392
  ordered pairs) with gravity, income, climate and demographic covariates.
- **`models/`** — a graph neural network whose climate features are gated by an
  income-conditioned FiLM layer (the *spatial gate*), plus PPML-gravity and
  random-forest baselines.

Everything here runs from public data. Nothing in the repository requires
restricted access; the one lab-access dataset that would improve the flow
measurement is documented in
[`data-pipeline/README.md`](data-pipeline/README.md) and is deliberately not
used.

---

## Reproducing the results

Three stages, in order. Stages 1 and 2 must be run before anything in `models/`.

```bash
git clone <repository-url>
cd Spatially-Gated_Deep_GNN
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 1. Build the panel

```bash
pip install -r data-pipeline/requirements.txt
cd data-pipeline
python src/00_download.py        # fetch raw inputs (~6 GB, mostly WorldClim)
make all                         # build the panel
make test                        # 149 unit tests
cd ..
```

Produces `data-pipeline/data/processed/`:

| file | rows | what it is |
|---|---:|---|
| `mx_dyadic.parquet` | 6,034,392 | the full panel, zeros explicitly present. Estimate on this. |
| `mx_dyadic_positive.csv` | 134,331 | the same 51 columns, positive flows only. Input to stage 2. |
| `mx_dyadic_codebook.md` | — | documents all 51 columns |
| `municipios_harmonized_2020.gpkg` | 2,457 | harmonized boundaries, keyed `cvegeo` |
| `municipios_centroids.csv` | 2,457 | centroids used to build the spatial graph |

### 2. Build the modelling splits

```bash
pip install -r models/requirements.txt
python models/make_splits.py
```

Writes `data/X_{train,test,inference}.csv`, the matching `y_*` and `y_*_log1p`
files, `data/muni_centroids.csv`, and `data/splits_manifest.md` recording the
row counts and settings actually used.

The split is over **source municipalities**, not over dyads — splitting dyads at
random would leak a municipality's covariates across the boundary. Municipalities
are stratified into GDP-per-capita terciles, 80/20; a dyad is `train` if both
endpoints are train municipalities, `test` if both are test, and `inference`
otherwise. With an 80/20 municipality split that gives roughly 64/4/32 percent of
pairs, which is geometry rather than a bug. `make_splits.py` asserts the
partition is exhaustive and that no test municipality appears in a training dyad.

### 3. Fit the models

```bash
python models/gravity.py                          # PPML gravity baseline
python models/random_forest.py                    # random-forest baseline
python models/gnn.py --name run1                  # train the GNN (800 epochs)
python data-pipeline/analysis/maps.py --name run1 # figures from that checkpoint
python data-pipeline/analysis/moran.py            # Moran's I + OLS spatial diagnostics
```

All of these write to `models/runs/` (except `moran.py`, which writes beside
itself) and accept `--data-dir` / `--outdir`. The two baselines redirect stdout to
a file by default; pass `--stdout` to watch them. `gnn.py --epochs N` shortens
training (the paper uses the default 800). On CPU an epoch is roughly 30 s at the
default width; `--gpu` uses CUDA if available.

`maps.py` is the figure generator. It lives under `data-pipeline/analysis/` for
historical reasons — that directory's README says nothing there is part of the
build, which is true of the *pipeline* build but no longer true of the paper's
figures. Moving it to `models/` would be tidier if you want the layout to match
the workflow.

---

## Layout

```
Spatially-Gated_Deep_GNN/
  data-pipeline/         panel construction — see its own README
    src/                 00_download.py … 08_income.py, plus shared modules
    tests/               149 unit tests
    reports/             generated build diagnostics
    analysis/
      maps.py            THE figure generator (reads a gnn.py checkpoint)
      moran.py           Moran's I + OLS spatial diagnostics
    config.yaml          every path, vintage and analytic choice
  models/
    make_splits.py       panel  ->  train/test/inference splits
    gnn.py               the Spatially-Gated Deep GNN: train + evaluate
    gravity.py           PPML gravity baseline
    random_forest.py     random-forest baseline
    requirements.txt
    runs/                generated: checkpoints, logs, figures (not versioned)
  data/                  modelling splits (generated; not versioned)
  CITATION.cff           TEMPLATE — complete before submission
  LICENSE                MIT
```

Derived data is not versioned. The panel files run to hundreds of megabytes and
are reproducible from `data-pipeline/`; see the comments in `.gitignore` for the
specific exceptions and why they exist.

## The model

`SpatialGatedDG` (in [`models/gnn.py`](models/gnn.py)) encodes municipalities
with a GCN layer followed by two GAT layers, and decodes ordered pairs from the
concatenated source embedding, destination embedding, their product, and a
projection of the edge features.

Two things make it more than a generic link-regression model:

- **The income-conditioned climate gate.** A small network maps each
  municipality's income features (`gdppc`, `gdppc_sq`) to FiLM parameters
  (γ, β) that scale and shift its climate features (`temp`, `precip`). Separate
  gates are learned for the source and destination roles, so the same
  municipality can respond to climate differently depending on which end of the
  flow it is on. The gate is what the paper is about; the fitted γ values are
  written to `models/runs/climate_embeddings_<name>.npz`.
- **A gravity skip connection**, initialised to unit weight on log distance, so
  training starts from a gravity baseline and learns departures from it.

Auxiliary reconstruction losses ask the node embeddings to predict both a
municipality's own features and its spatial lag, which keeps the embeddings
spatially informative.

## Verification status

| stage | status |
|---|---|
| `data-pipeline` tests | 149 passed |
| `models/make_splits.py` | runs; partition and leakage assertions pass |
| `models/gravity.py` | runs; elasticities 0.45 origin, 0.82 destination, −0.53 distance |
| `models/random_forest.py` | runs; OOB-tuned to `min_samples_leaf=50`, `n_estimators=200` |
| `models/gnn.py` | runs end to end on CPU; gate built over `temp`, `precip` |
| `analysis/maps.py` | runs; checkpoint loads, residual choropleth matched 452/452 municipios to geometry |
| `analysis/moran.py` | runs; **Global Moran's I = 0.3051** (p=0.001, n=1,959) on out-migration flows, 0.3199 on the OLS error; specification test selects SLM |

Statistical results have **not** been re-derived at full training length since
the port described below — the runs above were short smoke tests (1 and 20
epochs) whose purpose was to prove the code path, not to estimate anything.
Re-run stage 3 at the default 800 epochs and regenerate every figure and table
before submitting.

## Known issues to resolve before submission

These are real and are flagged rather than silently patched, because each is a
research decision rather than a bug with one correct answer.

0. **No model figures are committed — every one must be regenerated.** The
   repository previously carried eight figures in `models/gnn_figures/` that were
   output from the model's earlier **Brazilian** application: three drew the
   Brazilian national outline, and the gate plots labelled their x-axis "Mean
   Household Income (BRL)" — Brazilian reais. They have been deleted rather than
   left to be mistaken for results (they remain in git history).

   **If any figure in the manuscript was taken from that set, the paper shows the
   wrong country.** Regenerate all of them from a Mexican run — stage 3 above
   writes to `models/runs/figs_<name>/` — and check each one's axis units and
   geography by eye before it goes in.

1. **The committed analysis outputs predate the current panel.** Re-running
   `analysis/moran.py` against the freshly built panel moves the OLS fit from
   R² 0.5439 to 0.5388 on the same 1,958 municipalities. The split assignment is
   unchanged, so the covariate or flow values themselves have shifted — the
   pipeline has been rebuilt since those outputs were generated (the boundary
   crosswalk grew to 18 verified rows on 2026-07-22, which changes harmonization
   and therefore the flows). Every committed number and figure needs regenerating
   against one final panel build.

2. **Dependency pins disagree between the two halves.** `data-pipeline` pins
   numpy 2.1.3 / polars 1.17.1 / pandas 2.2.3 / geopandas 1.0.1; the models were
   verified against 2.3.3 / 1.41.2 / 2.3.3 / 1.1.0. Installing both files into
   one environment resolves to whichever came last, so no single environment has
   been verified against both. Reconcile the pins, or state that the two stages
   run in separate environments.

3. **`validation.published_internal_migrants_2015_2020` is still null**, so the
   pipeline has no external check on its headline migration total. Both
   `reports/flows_reconciliation.md` and `reports/assembly_validation.md` say so
   explicitly. See item 5 in `data-pipeline/README.md`.

## Decisions taken, and what they changed

Three questions were open when this package was assembled. All three are settled;
recorded here because each one moves numbers.

### Every municipality is kept

The original code dropped CVEGEO 21128 on a bare `!= 21128`. That filter is gone,
so the panel is the full 134,331 positive dyads rather than 134,301.

21128 (Puebla) is genuinely extreme — `gdppc` of 307,677 pesos against a median of
690, a Censos Económicos artifact where one large establishment's value added is
booked to a municipality of 8,771 people. But it sits at the top of a continuum
(26041 at 232,385; 04003 at 225,869; 27014 at 219,102), the old filter tested
`source_code` only so the same value survived on the destination side of 61 dyads,
and excluding the single most extreme observation while keeping the next three is
not a rule a reader can evaluate. The right response to the skew is the
specification — the panel provides `src_log_gdppc` — not deleting municipalities.

Effect: +30 dyads (train 86,384→86,409, inference 42,724→42,729, test unchanged).
The tercile cuts and the 1,962/491 municipality assignment are **unchanged**,
because the split was always computed before the filter was applied.

### `analysis/maps.py` is the figure generator; `models/gnn_map.py` is deleted

The two were forks of one script, writing the same filenames into the same
directory, so whichever ran last won. `maps.py` was the newer fork — it adds
`binned_line_plot()` and an `--income` flag, and its `climate_keywords` list had
already been fixed. `models/gnn_map.py` has been removed; the pre-port version is
still in git history if it is ever needed.

### The GNN's internal residual Moran's I is deleted, not repaired

`gnn.py` and `maps.py` each printed a residual Moran's I above 1 (`+3.40` at 1
epoch, `+2.00` at 20) — impossible under row-standardised weights. The cause was
`moran_on_subgraph()`: `adj_weights` is row-standardised over the **full**
2,457-node graph, the function then kept only edges with both endpoints in the
452-node test mask while reusing those weights (so the subgraph's `S0` fell far
below its node count), and it passed the full-length value array through unchanged
(so `N` was 2,457 and the mean and variance were taken over ~2,000 nodes not in the
subgraph). The `N/S0` factor was a full-graph *N* over a subgraph `S0`.

Both the function and the gravity-only least-squares baseline that fed it have
been removed rather than reimplemented. Spatial autocorrelation is reported from
[`analysis/moran.py`](data-pipeline/analysis/moran.py), which uses `esda.Moran` on
properly row-standardised weights: **Global Moran's I = 0.3051** (p=0.001,
n=1,959) on out-migration flows by source municipality, and 0.3199 on the OLS
error. `moran.py` now prints that figure to stdout as well as drawing it on the
map, so it does not have to be read off a PNG.

Keeping all municipalities does **not** move it: it was 0.3051 with 1,958
municipalities and is 0.3051 with 1,959.

The gate-activation Moran statistics in `gnn.py` and `maps.py` are retained
(`I≈0.26–0.35`). Those are computed over the whole graph, where `S0 == N` and the
mismatch does not arise.

## Notes for a reader coming from the Brazil version of this model

The model was originally applied to Brazilian data, and the port to Mexico left
several call sites behind. They are fixed, but they are worth knowing about
because each failed *silently* rather than raising:

- `import geobr` / `geobr.read_municipality()` — the Brazilian boundary package —
  supplied centroids and choropleth geometry. It would have attached Brazilian
  centroids to Mexican municipality codes. All geometry now comes from the
  pipeline's own harmonized boundary file, and there is no fallback: missing
  geometry is a hard failure.
- `climate_keywords` matched on a leading underscore (`"_temp"`, `"_precip"`),
  which matched none of this panel's prefix-stripped names (`temp`, `precip`).
  `climate_idx` came out **empty**, so the income-conditioned climate gate was
  built over zero features — the paper's central mechanism, silently disabled.
- `gravity_edge_idx` looked for `"pop_ratio"` and `"distance_km"`, neither of
  which exists here (distance is `dist_geodesic_km`), so the gravity skip
  connection became `nn.Linear(0, 1)`.
- **The training and figure scripts disagreed about what "income" means.**
  `gnn.py` matched `["gdp"]`; the figure script matched `["income"]`, which is the
  Brazil panel's `mean_income` column. Against this panel the second list matched
  nothing, so the two built structurally different gate networks from the same
  data and the checkpoint would not load. Both now use `["gdp"]`, and both
  hard-fail if either index list comes out empty — the guard that would have
  caught all of this years earlier.
- `gnn.py` loaded its best checkpoint unconditionally after training, but only
  ever wrote one inside an `epoch % 20` block, so any run shorter than 20 epochs
  threw `FileNotFoundError` after the training had already completed.
- Municipality codes were coerced to integers in several places. CVEGEO is a
  zero-padded 5-character string: `01001` becomes `1001` as an integer and every
  join against the boundary file then misses the nine states numbered 01–09.
  `data-pipeline/src/common.py` states the rule — codes are strings, forever —
  and the model code now follows it.
- Brasília (−15.78, −47.93) was the fallback centroid.

## Citation

See [`CITATION.cff`](CITATION.cff) — **currently a template**; the author list,
affiliations, ORCIDs, DOI and manuscript details still need filling in.

Data sources are cited in
[`data-pipeline/docs/references.bib`](data-pipeline/docs/references.bib).

## License

MIT — see [`LICENSE`](LICENSE). Note this covers the *code*; the underlying data
remains subject to the terms of its providers, chiefly
[INEGI](https://www.inegi.org.mx/inegi/terminos.html).
