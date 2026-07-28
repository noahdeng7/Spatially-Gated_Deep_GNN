# mx-migration

A reproducible pipeline that builds a **dyadic municipality-to-municipality
internal migration dataset for Mexico**, with gravity-model covariates attached
to each ordered origin-destination pair.

Outputs, both written by `07_assemble.py`:

| file | rows | what it is |
|---|---:|---|
| `data/processed/mx_dyadic.parquet` | 6,034,392 | the full panel: 2,457² minus the diagonal, **zeros explicitly present**. Estimate on this. |
| `data/processed/mx_dyadic_positive.csv` | 134,331 | same 51 columns, only the dyads with `migrants > 0`. The shareable deliverable — inspection, mapping, network analysis. **Not** for PPML; see [the zeros section](#the-zeros-are-in-the-panel-on-purpose). |

`data/processed/mx_dyadic_codebook.md` documents every one of the 51 columns;
`07_assemble.py` fails the build if a column reaches the panel without an entry.

---

## Status: all 10 inputs present (2026-07-21)

Every transformation, validation and report is written and tested (149 unit
tests). The four inputs previously marked MANUAL were recovered as follows:

| input | status |
|---|---|
| ITER municipal population (36 MB) | automated |
| WorldPop 2020 MEX 100m (28 MB) | automated |
| WorldClim tavg 30s (4.3 GB) | automated |
| WorldClim prec 30s (1.0 GB) | automated |
| Kummu gridded GDP admin-2 (114 MB) | automated |
| World Bank PPP + deflator | automated, values already in config |
| INEGI census microdata | **downloaded** — CA national file (see below: the full-count CB is lab-access only) |
| INEGI Marco Geoestadístico | **fallback** — ArcGIS Feature Service, Dec-2022 vintage (`fetch_mgn_arcgis.py`) |
| CONAPO municipal projections | **downloaded** — official RARs recovered via the Wayback Machine |
| Censos Económicos 2019 | **downloaded** — via the SAIC JSON API (`fetch_ce2019_saic.py`) |

```bash
pip install -r requirements.txt
python src/00_download.py --list     # status + confidence tier per input
python src/00_download.py --verify   # PASS/MISSING per manual input
make test
```

### The census microdata finding that changed a default

The full-count **Cuestionario Básico microdata is not publicly downloadable.**
The files on INEGI's microdatos page are explicitly *examples* — the page
states they "no permiten hacer ningún tipo de inferencia" and exist so users
can test syntax before submitting jobs to the **Laboratorio de microdatos** /
remote processing service. The public person-level instrument is the
**Cuestionario Ampliado ~10% sample**, national scope, one file:

```
https://www.inegi.org.mx/contenidos/programas/ccpv/2020/microdatos/Censo2020_CA_eum_csv.zip
```

(That URL is live; the previously-tested `..._CA_nal_csv.zip` variant never
existed — the national scope is abbreviated `eum`, and per-state files also
exist for most states.) `flows.use_basic_questionnaire` is therefore now
**false**: flows are FACTOR-weighted estimates from the sample, and the
full-count discussion below is retained as design documentation for anyone who
obtains the CB through the lab.

### One finding worth your attention

**INEGI serves missing files with HTTP 200 and an HTML error page**, not a 404:

```
HTTP 200 OK · Content-Type: text/html · 1428 bytes
"Esta liga ya no existe, lamentamos el inconveniente."
```

`raise_for_status()` sails straight past that. Uncaught, the downloader would
write that page to disk as `Censo2020_CA_nal_csv.zip`, `already_have()` would
report the input as PRESENT forever after, and the failure would surface three
steps later as a baffling parse error.

`00_download.py` now refuses any HTML response where a data file was expected,
and any download under 4 KB. Both have regression tests in
[test_download_guards.py](tests/test_download_guards.py). The general lesson,
which is why three entries are marked MANUAL rather than guessed: **do not trust
an INEGI URL you have not actually fetched.**

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/00_download.py          # fetch raw inputs (see TODOs)
make all                           # build everything
make test                          # unit tests
```

Every step is independently runnable and idempotent:

```bash
python src/03_distance.py          # just recompute distance
python src/03_distance.py --recompute-centroids
```

`make all` re-runs only what is stale. Row counts in and out of every step go to
stdout and to `logs/build.log`:

```
ROWS  aggregate to O-D cells [domestic]   in=11,938,402  out=486,331  delta=-11452071 (-95.93%)
JOIN  flows                               matched=486,331     unmatched=5,613,569   (7.97% matched)
```

---

## Layout

```
mx-migration/
  data/raw/          untouched downloads, never edited
  data/interim/      cleaned single-source tables (parquet)
  data/processed/    final dyadic panel + codebook
  src/
    common.py        logging, CVEGEO handling, interim IO
    geo.py           geometry loading, 2015→2020 crosswalk
    zonal.py         population-weighted zonal statistics
    climate_loader.py  ERA5-Land via Earth Engine (the default panel climate)
    00_download.py … 08_income.py
  tests/
  reports/           generated diagnostics (see below)
  analysis/          downstream plotting/Moran scripts — NOT part of the build
  config.yaml        every path, vintage and analytic choice
  Makefile
```

`common.py`, `geo.py` and `zonal.py` are a deviation from the requested file
list. They exist because the crosswalk is needed by four steps and the zonal
engine by two, and four private copies of "which municipalities changed" is how
a panel silently goes out of balance.

**`08_income.py` runs *before* `07_assemble.py`.** The number is its position in
the source tree, not in the build: income was added after the assembler existed,
and the assembler consumes `interim/income.parquet` like every other covariate.
`make all` gets this right from the prerequisite graph; `scripts/run_all.ps1`
orders `$steps` explicitly and resolves `-From` against that order rather than
against the numeric label.

**`climate_loader.py` is a build step, not a library.** `climate.panel_source`
defaults to `era5_gee`, so `interim/climate_gee.parquet` — not `05_climate.py`'s
`interim/climate.parquet` — is what reaches the panel. `05_climate.py` still runs
and still writes `reports/climate_diagnostics.md`, but under the default config
its output is a cross-check rather than an input.

### Generated reports

| file | what it tells you |
|---|---|
| `reports/flows_reconciliation.md` | where every person record went; buckets must sum to input |
| `reports/population_coverage.md` | which vintage is in force, missingness |
| `reports/centroid_containment.csv` | centroids falling outside their own polygon |
| `reports/gdp_source_comparison.md` | correlation between the two GDP constructions |
| `reports/climate_diagnostics.md` | distributions, out-of-range values |
| `reports/income_summary.md` | ICMM coverage, CV reliability bands, the split-child fold |
| `reports/demography_bands.md` | age bands in force and the specification deviation |
| `reports/orphans.csv` | flow codes absent from 2020 geometry (build stops if any) |
| `reports/assembly_validation.md` | every check, reconciliation, missingness by column |

---

## Decisions that materially affect results

Each has an implemented default. Each is a config flag. Each is here because a
reader of your results would want to know which way it was set.

### Boundary harmonization

**Default:** `geometry.crosswalk_strategy: aggregate_to_parent`

Mexican municipalities split. Several were created between 2015 and 2020 —
Morelos created three in 2017, Quintana Roo one in 2016, Baja California two in
2020. A person reporting 2015 residence in a municipality that has since been
carved up needs their code mapped onto the 2020 universe, or the join drops them.

Codes are mapped back to the pre-split **parent**, on **both** the origin and
destination side. The panel is balanced — the same municipal universe at both
ends of the window — at the cost of coarser geography in a handful of places. A
move from a split child to its own parent correctly lands on the diagonal, since
it was an internal move within the pre-split municipality.

*Alternative:* `allocate` — split the parent's flows across children by
population share. Implemented as a deliberate `NotImplementedError` rather than
silently falling back, because it manufactures within-parent variation the flow
data does not contain. If you want it, the honest version needs an external
source for the within-parent distribution.

*Alternative:* `none` — leave codes alone and let the orphan check fail. Useful
for diagnosing what actually changed.

> **⚠ The crosswalk ships unverified.** `data/raw/crosswalk_municipios_2015_2020.csv`
> is seeded from general knowledge of Mexican administrative changes, **not read
> off an INEGI source document**, and every row is marked `verified=False`. The
> CVEGEO codes in particular are the part most likely to be wrong. Verify each
> against the INEGI *Catálogo de claves de entidades federativas, municipios y
> localidades* before trusting a build.
>
> The safety net is that `audit_codes()` compares codes observed in the flows
> against codes present in the 2020 geometry in both directions, so a boundary
> change missing from the CSV shows up as an orphan and stops the build. An
> *omission* is loud. A *wrong code* would be silent — hence `verified=False`.

### Which GDP source

**Default:** `gdp.source: censos_economicos`

Both are always computed and correlated; the flag only picks which feeds the
canonical columns.

- **Censos Económicos 2019** — municipal gross value added ÷ 2020 population,
  deflated, converted at the World Bank PPP factor to 2017 international dollars.
- **Gridded (Kummu et al. / Wang & Sun)** — zonally aggregated to municipal
  polygons, population-weighted.

**If the two correlate below 0.80 (Pearson on logs), the build stops** with a
`DecisionRequired` and a diagnostic checklist, rather than silently picking one.
Threshold: `gdp.min_cross_source_correlation`.

**The gate fired on the first real build (2026-07-21)** — r(logs) = 0.5008,
Spearman = 0.4952, median CE/grid = 0.0645, n = 2,453 — and the checklist was
worked before accepting:

- **Not units** (A131A confirmed as millones; the ratio pattern is not a
  power of ten). **Not PPP/deflator** (cannot move r). **Not key alignment**
  (5-char joins, zero orphans; misalignment pushes r toward 0).
- **Not circular** — this resolves old TODO #8: Kummu band 30 sampled at the
  2,464 population-weighted centroids shows genuine within-state municipal
  variation (median 43 distinct values per state, ≈1 per municipality in most
  states; median within-state CV 0.32). The cross-check is a real cross-check.
- The residual disagreement is the documented measurement difference:
  **censal VA excludes informal and subsistence activity** and books value
  added at establishment location. CE municipal GDP pc median is 679 int$
  against the grid's 11,323 int$ — the gap is the informal economy, and it is
  largest exactly in poor rural origins.

**Decision:** keep `gdp.source: censos_economicos` (a direct measurement with
a known, documented bias, rather than a modeled surface), ship `gdppc_grid`
alongside as always, and lower `gdp.min_cross_source_correlation` to 0.45 —
just under the observed value, so a *future* regression still halts the build.
Anyone estimating with the CE-based `gdppc` should read
`reports/gdp_source_comparison.md` first and consider `gdppc_grid` as a
robustness check; the two will produce different coefficients.

### Origin population vintage

**Default:** `population.origin_pop_year: 2015`

2015 population is the stock at risk of emigrating during the window. The 2020
figure is mechanically *depleted* by the very flow being modelled, which puts the
dependent variable on both sides of the equation.

Both `pop_orig_2015` and `pop_orig_2020` are always emitted; the flag only sets
which one `pop_orig` aliases, and `pop_orig_vintage` records it in the data.
Switching is a config change and a re-run of step 07, not a re-extraction.

### Unspecified and international origins

**Default:** excluded from the dyadic panel, retained and counted in
`interim/flows_excluded.parquet` and `reports/flows_reconciliation.md`.

Records are routed into exactly one bucket — `domestic`, `non_migrant`,
`foreign`, `not_specified`, `under_min_age`, `unparseable_code` — and the buckets
are asserted to sum back to the input. Nothing is deleted. "How many arrived from
abroad" is a real number someone will ask for.

*Alternative:* impute unspecified origins proportionally to observed flows. Not
done: it would inflate the very cells that are already best measured and
manufacture precision in exactly the wrong direction.

### Climate product

**Default:** `climate.source: worldclim` (v2.1, 30 arc-second, 1970–2000 normals)

Most widely used in the gravity literature, so coefficients are comparable
across studies. `chelsa` (v2.1, better orographic downscaling — relevant for
Mexico's sierras, less standard) and `terraclimate` (4 km, covers 2010–2020, much
closer to the migration window) are both configurable.

> **These are long-run normals, not window conditions.** They identify the effect
> of *climate* — a persistent locational attribute people sort on — not *weather
> shocks*, a transitory push factor. Different question, different identification.
> If you want shock-driven displacement, switch to `terraclimate`.
>
> TerraClimate ingestion is scaffolded with a specific `NotImplementedError`
> describing what it needs: mean temperature must be derived as (tmax+tmin)/2
> because TerraClimate publishes no tmean band, and the order of within-year vs
> across-year aggregation matters for precipitation.

`climate.source` selects the product `05_climate.py` builds. Which table actually
feeds the panel is the separate `climate.panel_source`, defaulting to `era5_gee`
(`src/climate_loader.py`): ERA5-Land daily reanalysis averaged over 2016–2020,
reduced server-side on Google Earth Engine. That is *window* climate rather than
a long-run normal — conditions during the migration window.

**Every climate path is population-weighted.** Temperature and precipitation are
intensive: they do not add across space, so an unweighted polygon mean answers
"what is the average condition over this territory", letting 40,000 km² of empty
Sonoran desert outvote the town where everyone lives. Both the WorldClim zonal
path and the ERA5/GEE path weight by the WorldPop 2020 grid — the same grid
behind the population-weighted centroids — so distance and climate refer to one
consistent notion of where people are. The ERA5 path computes
`Σ(value·pop)/Σ(pop)` at ERA5-Land's native ~11 km scale (population aggregated
*up* to the climate grid, never climate resampled down), returns the unweighted
mean alongside so the size of the correction is auditable, and flags in
`climate_weighting` the handful of municipalities carrying no population under
the ERA5 land mask, which fall back to the unweighted mean rather than silently
blending.

### Youth share is municipal, not national

The variable was specified as youth share of the origin **country**. For an
internal-migration panel every origin is Mexico, so that is a constant: no
variance, absorbed by any origin fixed effect, identifies nothing.

Built at **origin municipality** level instead. Bands are config
(`demography.youth_min_age` etc.), defaulting to 15–29 over 15–64, because the
literature is not settled — ILO uses 15–24, the migration literature often uses
20–34, and the choice moves the variable. The raw numerator and denominator ship
alongside the ratio so it can be redefined without re-extraction.

---

## Things worth knowing before you use the output

### Municipal GDP per capita is constructed, not official

Mexico publishes no official municipal GDP series; ITAEE stops at the state
level. The construction is documented in `reports/gdp_source_comparison.md`.

**The bias that matters:** censal value added covers economic-census
establishments and systematically understates subsistence agriculture and
informal activity. Those concentrate in poor rural municipalities — exactly the
high-emigration origins the gravity model is most sensitive to. So the measure is
biased downward *precisely where it matters most*, and the bias is correlated
with the dependent variable rather than random.

### Household income is modelled, and it is a mean, not a median

`src_hh_income` / `dst_hh_income` come from INEGI's **ICMM 2020** (*Ingreso
Corriente para los Municipios de México*, released October 2023). The Mexican
census does not ask household income, so no observed municipal income exists
anywhere; ICMM fills the gap by **small-area estimation** — a model fit on ENIGH
2020 (nueva serie) with auxiliaries from the 2020 Census and administrative
records, applied to every municipality. Three things follow:

- It is **modelled, not measured**. Judge any single municipality's value with
  `*_hh_income_cv`; INEGI's convention is <15% reliable, 15–30% caution, >30%
  not recommended. Every municipality in the 2020 vintage is under 9%.
- It is a **mean**, not a median — INEGI publishes no municipal median, and
  small-area models target the mean.
- Units are as published: **quarterly, nominal 2020 pesos, per household**, with
  no PPP conversion. That makes its *level* not comparable to `src_gdppc`, which
  is 2017 international dollars. Multiply by 4 for annual pesos.

The one construction choice worth knowing is the split-child fold. Income is
**intensive**, so a post-2015 child folded back into its parent is
population-weighted *averaged*, not summed —
`M_parent = (H_p·M_p + H_c·M_c)/(H_p + H_c)`, with 2020 population standing in
for household counts, which ICMM does not publish. In the 2020 vintage this
touches exactly one municipality: 23005 Benito Juárez absorbing 23011 Puerto
Morelos. Folded rows are flagged `icmm_2020_harmonized` and their se/ci/cv are
set NA, because a weighted mean of two SAE point estimates carries no clean
combined standard error. See `reports/income_summary.md`.

### The zeros are in the panel on purpose

Roughly 6.1M rows, of which a small fraction have positive flow. Dropping zeros
and running OLS on `log(flow)` conditions on the dependent variable being
positive, biasing the distance elasticity toward zero by an amount that depends
on how many cells were dropped. PPML needs the zeros present.

`flow_observed` marks pairs that appear in the sample at all. `zero_class` is a
**heuristic, not an identified quantity** — with one 10% cross-section,
structural and sampling zeros are not separately identifiable. A true flow of 4
people has roughly a two-thirds chance of contributing zero sampled records. Use
it for sensitivity analysis, not as a covariate.

### Flows are built from the CA 10% sample (the full count is not public)

`flows.use_basic_questionnaire: false` — **not by choice, by availability.**
The design below originally targeted the Cuestionario Básico full count, and
the pipeline still supports it, but the CB microdata turned out to be
distributed only through INEGI's Laboratorio de microdatos (the public "CB"
files are explicit examples). What is public is the CA sample, and `FACTOR` is
**confirmed present** in the real `Personas00.CSV` header of the national CA
file, along with all five variables this pipeline uses (`ENT`, `MUN`, `EDAD`,
`ENT_PAIS_RES_5A`, `MUN_RES_5A`).

Consequences of sample mode, stated plainly:

- `migrants` is a **FACTOR-weighted estimate**, not a headcount.
  `migrants_unweighted` (raw record count) ships alongside it.
- **Sampling zeros are real.** A dyad with a true flow of 4 people has roughly
  a two-thirds chance of contributing no sampled records; the `zero_class`
  heuristic in `07_assemble.py` matters again, with all the caveats it carries.
- Denominators carry sampling error.

If you obtain the CB through the Microdata Lab, drop its person tables under
`data/raw/censo2020/` (they must NOT carry the `_CA` filename marker), flip the
flag, and everything above reverts to the full-count behaviour — the split on
the `_CA` marker in `01_flows.py` keeps the two instruments from ever being
read together, which would double-count every sampled dwelling.

### Want a positive-flows-only file?

`assemble.export_positive_only` (on by default) writes
`data/processed/mx_dyadic_positive.csv` alongside the full panel — same columns,
same construction, only the dyads with `migrants > 0`. Good for inspection,
mapping, network analysis, or any tool that will not hold several million rows.
The full panel is untouched.

Two things to know:

**"Drop the structural zeros" and "every dyad has positive flow" are different
operations.** On a synthetic panel with realistic sparsity, dropping only the
zeros classified `structural_zero_likely` leaves 86,435 rows — of which 85,322
are *still zeros*. Nearly every zero is `sampling_zero_plausible`. There is
deliberately no config option to filter on `zero_class`, because that heuristic
cannot bear the weight: it would delete real flows while looking principled.
The export filters on `migrants > 0`, which is unambiguous.

**If you want the main panel itself filtered**, set
`assemble.drop_zero_flows_from_main_panel: true`. That is a real option and
sometimes the right one — but it drops ~99% of rows and makes the file unusable
for PPML. When set, `07_assemble.py` logs a prominent warning and the codebook
opens with an unmissable banner saying the panel is filtered and should not be
used for gravity estimation. A filtered file tends to outlive the conversation
that produced it. Reverting is a config change plus a re-run of step 07 only —
nothing upstream rebuilds.

### Distance uses population-weighted centroids

Geometric centroids would bias the distance-decay term specifically in large
sparse northern municipalities — Ensenada is ~52,000 km² with its population
pinned to one corner, and its geometric centroid sits in empty desert far from
anyone. That is bias, not noise, and it is concentrated in high-emigration units.

Weights come from the WorldPop 2020 100 m grid — the same grid used for the
climate weighting, so distance and climate refer to a consistent notion of where
people are. Distance is geodesic on WGS84 (Karney, via `pyproj.Geod`); the
great-circle measure is retained alongside because the literature is split.

The diagonal gets the Head–Mayer internal distance (0.67·√(area/π)) even though
it is excluded from the main panel.

### The flow variable is a 5-year stock question

The census asks where you lived five years ago, not how many times you moved.
Return migrants who left and came back appear as non-migrants; multiple moves
collapse to one origin-destination pair; people who died or emigrated abroad
before the census are absent entirely.

---

## Design rules this pipeline follows

**Geographic codes are strings, forever.** `CVEGEO` is 2-digit state + 3-digit
municipality, zero-padded, `string` dtype end to end. Interim tables are parquet
rather than CSV specifically because a CSV round-trip is where `"01001"` turns
back into `1001`. `assert_cvegeo_valid()` guards every step. There are tests for
this because the failure is silent: an integer cast eats Aguascalientes' leading
zero and you get a smaller but entirely plausible panel.

**No inner joins.** Every join is `LEFT` with mandatory match reporting. An inner
join deletes non-matching rows silently, and a silent inner join is the single
most likely way this pipeline would produce quiet nonsense. Orphan codes stop the
build (`validation.fail_on_orphans`).

**Every raster's CRS is verified explicitly** before any zonal operation, and an
unlabelled CRS is refused rather than assumed. A wrong-but-plausible CRS produces
zonal statistics that look entirely reasonable and describe the wrong places.

**Weighted means are asserted to lie within the range of their own cells.** That
is arithmetically guaranteed for non-negative weights, so if it trips, nodata has
leaked into the weight array or two grids are misaligned. The assertion is not
disableable from config for a reason.

**Out-of-range values are reported, never clipped.** Clipping a 300 °C
temperature hides CHELSA's K×10 integer encoding instead of revealing it.

**Nothing is dropped, only routed and counted.** Every person record lands in
exactly one bucket and the buckets are asserted to sum back to the input.

**The finer grid is aggregated up to the coarser one.** Resampling 4 km climate
to 100 m does not create 100 m information — it creates 1,600 copies of each
value and lends them the appearance of precision that downstream standard errors
will treat as real. Population is aggregated with `sum` (extensive); intensive
variables get population-weighted means.

**Every column in the output has a codebook entry.** `07_assemble.py` fails the
build if a column reaches the panel without one.

---

## Open TODOs

Things that need a human, collected in one place. Resolved items are struck
through so the list doubles as a record.

1. ~~Fetch the four manual inputs~~ — **done 2026-07-21, all four.**
   - *Census microdata*: `Censo2020_CA_eum_csv.zip` (national CA sample,
     486 MB) — the full-count CB is Microdata-Lab-only, see the section above.
   - *Marco Geoestadístico*: ArcGIS Feature Service fallback
     (`src/fetch_mgn_arcgis.py`, Dec-2022 vintage, crosswalk folds the six
     post-census municipalities back). The official 2020 portal download
     remains preferable if it ever resurfaces.
   - *CONAPO*: the official `base_municipios_final_datos_{01,02}.rar` were
     recovered byte-identical from the **Wayback Machine** (the live link
     404s); merged into `proyecciones_municipales.csv`, 2,457 municipios,
     2015 total 121,347,800.
   - *Censos Económicos*: SAIC turns out to have a **JSON API** behind the
     interactive app — `src/fetch_ce2019_saic.py` drives it and wrote VACB
     for 2,463 of 2,478 municipios (the missing 15 are post-CE2019 creations
     and confidentiality suppressions). National total 9,983,798 millones de
     pesos, matching the published CE 2019 figure.
2. ~~Verify the crosswalk codes~~ — **done, and it mattered.** Checked against
   INEGI ITER 2020; the seeded table was badly wrong. `17020` is **Tepoztlán**,
   not Puente de Ixtla, and `17026` is **Tlayacapan**, not Tetela del Volcán —
   so the original rows would have folded two unrelated municipalities' flows
   into split children on the other side of the state. The three Morelos child
   names were also permuted. Corrected mapping, all five rows now `verified`:

   | child | | parent |
   |---|---|---|
   | 17034 Coatetelco | ← | 17015 Miacatlán |
   | 17035 Xoxocotla | ← | 17017 Puente de Ixtla |
   | 17036 Hueyapan | ← | 17022 Tetela del Volcán |
   | 23011 Puerto Morelos | ← | 23005 Benito Juárez |
   | 02006 San Quintín | ← | 02001 Ensenada |

   San Felipe (`02007`) was **removed**: ITER lists Baja California with exactly
   six municipalities in 2020 and no San Felipe, so it is not in the census
   universe and no 2015-residence code can refer to it.

   Still not machine-verified: the parent–child *relationship* itself. ITER
   confirms the codes and names but does not record lineage. Once the Marco
   Geoestadístico is available, adjacency becomes checkable — a genuine parent
   must border its child.

   **Update 2026-07-22 — crosswalk now 18 verified rows.** Seven more
   window-created municipalities were found (they surfaced as NA blocks in the
   panel): 04012 Seybaplaya and the six Chiapas creations 07120–07125. Parents
   were verified three ways: INEGI's own decree table
   (`censo2020_cpv_nuevos_municipios_a.pdf`), an empirical locality match
   (every 2020 locality of each child matched to its ITER **2010** municipality
   by coordinates — ≥99.3% of population from a single donor in every case),
   and polygon adjacency. The one non-adjacent pair (Capitán Luis Ángel Vidal /
   Siltepec) is explained: Honduras de la Sierra (2019, same parent) was carved
   from the land between them, and the three-way union reconstructs pre-2017
   Siltepec as a single polygon. The harmonized universe is now **2,457** — 
   exactly the 2015 Intercensal universe, which is what a balanced
   2015→2020 panel requires.
3. ~~`gdp.ppp.conversion_factor`~~ — **done**: `8.913552` (World Bank
   `PA.NUS.PPP`, Mexico, 2017).
4. ~~`gdp.ppp.deflator_from_year_to_base`~~ — **done**: `0.911176`, derived from
   the World Bank GDP deflator (2017: 95.053, 2019: 104.319; index base
   2018 = 100). Used in preference to INPC because it is API-accessible and
   therefore reproducible, and because a GDP deflator is the more appropriate
   index for a value-added series. **But see item 7.**
5. **`validation.published_internal_migrants_2015_2020`** — still null. The
   national 5-year internal migrant total from INEGI's published migration
   tabulado. Until it is set, the pipeline has no external check on its headline
   number, and both the reconciliation report and `assembly_validation.md` say
   so explicitly.
6. ~~Confirm `flows.vars.factor`~~ — **done**: `FACTOR` is confirmed directly
   in the header of the real national CA person file (stronger than the
   descriptor). All five analysis variables confirmed in the same header.
   `flows.special_codes` **still remains unverified** — the sentinels for "no
   especificado" and the foreign-country threshold cannot be checked
   automatically and are the likeliest source of a misclassified bucket.
7. ~~Confirm the Censos Económicos reference year~~ — **done: it is 2018.**
   The SAIC API indexes the quinquennial censuses by *reference* year
   (2003/2008/2013/2018/2023); CE 2019 answers to `anios=[2018]`. The deflator
   is now `0.950530` (2018→2017). Also settled on the way: the VACB variable
   (`A131A`) is denominated in **millones** de pesos, not miles —
   `gdp.censos_economicos.units` is `millions_of_pesos`.
8. **Check whether Mexico has genuine admin-2 data in the Kummu product.** If
   its Mexican municipal values were interpolated from state totals rather than
   built from Mexican subnational sources, the GDP "cross-check" is largely
   circular — it would agree with itself rather than validate INEGI. Confirm
   before reporting the correlation as validation.

### A design question the research opened up

The 5-year migration question (`ENT_PAIS_RES_5A`, `MUN_RES_5A`) appears in the
descriptor for the **basic** questionnaire, not only the expanded one. If that
holds, flows could be built from the **full count** rather than the 10% sample:
no sampling error, no expansion factor, far fewer sampling zeros — which would
also sharpen the `zero_class` distinction that `07_assemble.py` currently has to
treat as a weak heuristic.

The cost is losing the CA-only covariates, which this pipeline does not use.
`flows.use_basic_questionnaire` is stubbed in config for this. Worth confirming
before committing to the sample; it would be a strictly better spine.
