"""End-to-end runner for the standalone dyad pipeline.

  python run_pipeline.py                 # run everything
  python run_pipeline.py --no-impute     # skip the IDW imputation step
  python run_pipeline.py --no-encode     # skip top_crop one-hot (currently a no-op; see step 5)
  python run_pipeline.py --features X,Y  # subset (comma-separated loader names)
  python run_pipeline.py --clear-cache   # wipe data/intermediate/ before running

Layout: this runner sits in the pipeline root; feature loaders live in
feature_loaders/, the shared helpers (utils, dyad, features, census,
download_data, data_manifest) in helpers/, and all data in data/{raw,
intermediate,final}. Both code folders are put on sys.path below so the
existing flat imports (`from utils import …`, `import agriculture_loader`) work.

Data flow:
  1. census.ALL_CENSUS  → 8 census features from the censobr 2010 parquet.
  2. features.CANONICAL_LOADERS → 14 loaders for everything else, each pulling
     from the rawest institutional source (IBGE SIDRA, ERA5, MODIS, MapBiomas,
     HOTOSM, FUNAI/INSA/BATER). No external/sibling-folder dependency. All five
     climate loaders live in climate_loader (GEE, no Copernicus CDS) and are a
     2005–2010 average of the annual metric (per-year results averaged per muni,
     each year checkpointed; see utils.average_over_years). They produce exactly:
     wind_mean; wet_bulb_F; uv_log_mean; num_degreedays/degreedays_streak;
     and temp/precip/z_from_baseline_temp/z_from_baseline_precip (climate_baseline:
     2005–2010 mean vs a 1980–2000 baseline — the slowest single step).
  3. dyad.merge_features → one wide per-muni table.
  4. dyad.impute → per-column imputation:
       NDVI → IDW-KNN haversine (k=8, p=2); urban_area_pct → state-mean;
       9 climate cols → per-column median; everything else → zero.
  5. dyad.encode → one-hot expand top_crop_2010 into crop_* (no-op: crop_economics()
     no longer emits top_crop_2010, so this step currently does nothing).
  6. features.flows_edgelist_from_parquet → O-D edgelist from V6264.
  7. dyad.build_dyad → mirror src_/dst_ + haversine + log_flow + *_diff + same_state.
  7b. dyad.select_final → project onto the agreed final variable whitelist
      (each per-muni var mirrored src_/dst_, dyad-level vars once; drops the rest).
  8. Final CSV → data/final/dyad_dataset.csv.

Each producer's output is checkpointed to data/intermediate/*.parquet so a crash
or OOM mid-loop preserves earlier work — rerun and you'll see CACHE lines.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

# Windows consoles default to cp1252; our progress output (and this docstring
# via --help) uses unicode (→, —), so force UTF-8 to avoid UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# Make helpers/ and feature_loaders/ importable as flat modules before any
# project import below (they use `from utils import …`, `import <loader>`).
_ROOT = Path(__file__).resolve().parent
for _sub in ("helpers", "feature_loaders"):
    _p = str(_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import census
import dyad
import features
from utils import INTERMEDIATE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-census", action="store_true",
                   help="Skip the censobr parquet census step.")
    p.add_argument("--no-impute", action="store_true",
                   help="Skip the per-column imputation step.")
    p.add_argument("--no-encode", action="store_true",
                   help="Skip top_crop one-hot encoding.")
    p.add_argument("--features", type=str, default=None,
                   help="Comma-separated subset of feature loaders to run.")
    p.add_argument("--no-cache", action="store_true",
                   help="Ignore intermediate/ parquet cache; always recompute.")
    p.add_argument("--clear-cache", action="store_true",
                   help="Delete intermediate/ cache before running.")
    return p.parse_args()


def _run(name: str, fn, cache: Path | None = None, use_cache: bool = True):
    if cache and use_cache and cache.exists():
        df = pd.read_parquet(cache)
        print(f"  CACHE {name:24s}  {len(df):>6} rows  (← {cache.name})", flush=True)
        return df

    print(f"  RUN   {name:24s}  ...", flush=True)
    t0 = time.time()
    try:
        out = fn()
        rows = len(out) if out is not None else 0
        if cache and out is not None and not out.empty:
            cache.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(cache, index=False)
        print(f"  OK    {name:24s}  {rows:>6} rows  ({time.time() - t0:5.1f}s)", flush=True)
        return out
    except FileNotFoundError as e:
        print(f"  SKIP  {name:24s}  missing input: {e}", flush=True)
        return None
    except Exception as e:
        print(f"  FAIL  {name:24s}  {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return None


def main() -> int:
    args = parse_args()

    if args.clear_cache and INTERMEDIATE.exists():
        for f in INTERMEDIATE.glob("*.parquet"):
            f.unlink()
        print(f"Cleared cache: {INTERMEDIATE}")

    use_cache = not args.no_cache

    print("=" * 64)
    print("  Brazil dyad dataset pipeline — standalone")
    print("=" * 64)

    tables: dict[str, "any"] = {}

    # 1. Census features (censobr parquet).
    if not args.skip_census:
        print("\n[Census] censobr 2010 parquet")
        for name, fn in census.ALL_CENSUS.items():
            df = _run(f"census/{name}", fn,
                      cache=INTERMEDIATE / f"census_{name}.parquet",
                      use_cache=use_cache)
            if df is not None:
                tables[f"census_{name}"] = df
            gc.collect()

    # 2. Everything else (raw institutional sources).
    print("\n[Features] raw-source loaders")
    chosen = set(args.features.split(",")) if args.features else None
    for name, fn in features.ALL_LOADERS.items():
        if chosen and name not in chosen:
            continue
        df = _run(name, fn,
                  cache=INTERMEDIATE / f"feature_{name}.parquet",
                  use_cache=use_cache)
        if df is not None:
            tables[name] = df

    if not tables:
        print("\nNothing loaded — check raw_data/ and the producer setup. Aborting.")
        return 1

    # 3. Merge.
    print(f"\n[Merge] {len(tables)} feature tables")
    master = dyad.merge_features(tables)
    print(f"  master shape: {master.shape[0]} munis × {master.shape[1] - 1} feature cols")

    # 4. Impute.
    if not args.no_impute:
        print("\n[Impute] per-column (NDVI=IDW-KNN, urban_area_pct=state-mean, "
              "climate=median, rest=zero)")
        master = dyad.impute(master)

    # 4b. Accessibility (population gravity potential; needs all munis at once).
    print("\n[Accessibility] population gravity potential")
    master = dyad.add_accessibility(master)

    # 4c. Population density (persons per km^2; needs muni polygon areas).
    print("\n[Density] population density (pop / area_km2)")
    master = dyad.add_pop_density(master)

    # 4d. GDP per capita (SIDRA total GDP / census population).
    print("\n[GDP] GDP per capita (gdp_brl_2010 / total_pop_2010)")
    master = dyad.add_gdp_per_capita(master)

    # 5. Encode.
    if not args.no_encode:
        print("\n[Encode] top_crop one-hot")
        master = dyad.encode(master)

    # 6. Flow edgelist.
    print("\n[Flows] migration edgelist (V6264 unweighted)")
    flows = _run("flows_edgelist_from_parquet",
                 features.flows_edgelist_from_parquet,
                 cache=INTERMEDIATE / "flows_edgelist_from_parquet.parquet",
                 use_cache=use_cache)
    if flows is None or flows.empty:
        print("No flows — cannot build dyad. Writing per-muni master only.")
        out = dyad.save_final(master, "muni_features_only.csv")
        print(f"\nWrote per-muni features → {out}")
        return 1
    print(f"  edgelist: {flows.shape[0]} O–D pairs")

    # 7. Dyad construction.
    print("\n[Dyad] mirror src_/dst_ + pair features")
    d = dyad.build_dyad(master, flows)
    print(f"  dyad shape: {d.shape[0]} rows × {d.shape[1]} cols")

    # 7b. Project onto the agreed final variable set (drop stray mirrored cols).
    print("\n[Select] project onto final variable whitelist")
    d = dyad.select_final(d)
    print(f"  final shape: {d.shape[0]} rows × {d.shape[1]} cols")

    out = dyad.save_final(d)
    print(f"\nFINAL: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
