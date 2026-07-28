"""
Build the modelling splits from the assembled dyadic panel.

This is the bridge between `data-pipeline/` (which produces the panel) and
`models/` (which estimate on it). Before this script existed, the model scripts
read `data/X_train.csv` and friends, and nothing in the repository produced
them -- the reproduction chain had a hole in the middle.

Input  : data-pipeline/data/processed/mx_dyadic_positive.csv   (134,331 dyads)
Output : data/X_{train,test,inference}.csv
         data/y_{train,test,inference}.csv          (migrants, flow space)
         data/y_{train,test,inference}_log1p.csv    (log1p(migrants))
         data/muni_centroids.csv                    (copied from the pipeline)
         data/splits_manifest.md                    (row counts + settings)

Run from anywhere:

    python models/make_splits.py

## The split is over municipalities, not over dyads

Splitting dyads at random would leak: municipality *i*'s population, GDP per
capita and climate would appear in both train and test through every dyad it
participates in, and the model would be scored on covariates it had already
fit. So the split is assigned to **source municipalities**, stratified into
income terciles by `src_gdppc`, and a dyad is:

  - train      if BOTH endpoints are train municipalities
  - test       if BOTH endpoints are test municipalities
  - inference  otherwise (one endpoint train, one test)

The `inference` bucket is the train/test *boundary* -- it is neither a clean
training set nor a clean held-out set, and it exists so that the partition is
exhaustive and no dyad is silently dropped. It is much the largest of the three;
that is geometry, not a bug (with a 80/20 municipality split, the fraction of
pairs that are mixed is 2*0.8*0.2 = 32% of all ordered pairs, against 64% pure
train and 4% pure test).

Municipalities that never appear as a source are in neither `train_cities` nor
`test_cities`, so every dyad they receive lands in `inference`.

## No municipality is excluded

Every municipality in the panel is retained. The original code dropped CVEGEO
21128 on a bare `!= 21128`; that filter has been removed. See the comment on
`WHITELIST_FEATURES` below for what it was reacting to and why keeping the
municipality is the better call.

## Geographic codes are strings

`data-pipeline/src/common.py` states the project rule: "geographic codes are
strings, forever, and the moment one becomes an integer the leading zero on
Aguascalientes (01) disappears and joins start failing silently." The panel is
written to CSV, and a naive `pl.read_csv` infers `src`/`dst` as Int64, turning
`01001` into `1001`. Every downstream join against the harmonized boundary file
(whose `cvegeo` is a zero-padded string) then misses the nine states numbered
01-09. This script reads and writes both keys as 5-character strings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

# Repo root = parent of models/
ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PANEL = ROOT / "data-pipeline" / "data" / "processed" / "mx_dyadic_positive.csv"
DEFAULT_CENTROIDS = ROOT / "data-pipeline" / "data" / "processed" / "municipios_centroids.csv"
DEFAULT_OUTDIR = ROOT / "data"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_STRATA = 3

# The identifier columns as the models expect to find them. The panel calls them
# `src`/`dst`; every model script calls them `source_code`/`dest_code`.
ID_RENAME = {"src": "source_code", "dst": "dest_code"}

# --- no municipality is excluded --------------------------------------------
# EVERY municipality in the panel is kept. The original modelling code carried a
# bare `!= 21128` filter with no comment; it has been removed deliberately.
#
# What that filter was reacting to is real and worth knowing when reading the
# results. CVEGEO 21128 (Puebla) carries src_gdppc = 307,677 pesos -- the maximum
# in the panel, against a median of 690 and a 99th percentile of 37,572. It is a
# Censos Economicos artifact: a municipality of 8,771 people hosting an
# establishment whose value added is booked to it in full, so per-capita GDP is a
# ratio of a national-scale numerator to a village-scale denominator. Several
# other municipalities sit on the same heavy right tail (26041 at 232,385; 04003
# at 225,869; 27014 at 219,102), so 21128 is the extreme of a continuum rather
# than a separated point.
#
# It is kept for three reasons. Dropping it was never principled: the filter
# tested `source_code` only, so 21128 was removed as an origin (30 dyads) but
# retained as a destination (61 dyads) where `dst_gdppc` carried exactly the same
# value. Nothing about the tercile strata or the fitted models requires its
# removal. And excluding the single most extreme observation, while keeping the
# next three, is not a rule a reader can evaluate.
#
# The GDP-per-capita distribution is heavily right-skewed either way; that is a
# property of the Censos Economicos denominator and should be addressed in the
# specification (the panel provides `src_log_gdppc` for this) rather than by
# deleting municipalities.

# Features handed to the models. Kept deliberately narrow: the gravity terms
# (population, distance), the income terms the spatial gate is defined over
# (GDP per capita and its square), climate (temperature, precipitation), and
# demographic pressure. Source and destination must appear symmetrically -- the
# assertion below enforces that, because an asymmetric feature set would let the
# model distinguish origin from destination on feature availability alone.
WHITELIST_FEATURES = [
    "source_code", "dest_code",
    "src_pop", "dst_pop",
    "dist_geodesic_km",
    "src_gdppc", "src_gdppc_sq",
    "dst_gdppc", "dst_gdppc_sq",
    "src_temp", "src_precip",
    "dst_temp", "dst_precip",
    "src_dempres", "dst_dempres",
]

TARGET = "migrants"


def assert_src_dst_symmetric(features: list[str]) -> None:
    """Every `dst_x` must have an `src_x` and vice versa."""
    src = {f[4:] for f in features if f.startswith("src_")}
    dst = {f[4:] for f in features if f.startswith("dst_")}
    if src != dst:
        raise ValueError(
            "feature set is not symmetric between source and destination:\n"
            f"  src-only: {sorted(src - dst)}\n"
            f"  dst-only: {sorted(dst - src)}"
        )


def load_panel(path: Path) -> pl.DataFrame:
    """Read the panel, forcing the two geographic keys to zero-padded strings."""
    df = pl.read_csv(
        path,
        schema_overrides={"src": pl.Utf8, "dst": pl.Utf8},
        infer_schema_length=10_000,
    )
    df = df.rename(ID_RENAME)
    for col in ("source_code", "dest_code"):
        df = df.with_columns(pl.col(col).str.zfill(5).alias(col))
        bad = df.filter(pl.col(col).str.len_chars() != 5).height
        if bad:
            raise ValueError(f"{bad} rows have a {col} that is not 5 characters")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", type=Path, default=DEFAULT_PANEL,
                    help="assembled positive-flow panel (default: pipeline output)")
    ap.add_argument("--centroids", type=Path, default=DEFAULT_CENTROIDS,
                    help="municipality centroids written by 03_distance.py")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                    help="where to write the splits (default: <repo>/data)")
    ap.add_argument("--test-size", type=float, default=TEST_SIZE)
    ap.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = ap.parse_args()

    assert_src_dst_symmetric(WHITELIST_FEATURES)

    if not args.panel.exists():
        raise SystemExit(
            f"panel not found: {args.panel}\n"
            "Build it first:  cd data-pipeline && make all"
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def report(msg: str) -> None:
        print(msg)
        lines.append(msg)

    report(f"panel        : {args.panel.relative_to(ROOT)}")
    df = load_panel(args.panel)
    report(f"panel rows   : {df.height:,}")

    # --- stratify source municipalities into GDP-per-capita terciles ---------
    # The strata, the tercile cuts and the train/test assignment are computed over
    # the whole panel. This matches the original modelling code, which also
    # stratified and split before applying its (now removed) exclusion filter, so
    # the tercile cuts and the municipality assignment are unchanged by dropping
    # that filter -- only the 30 dyads it used to remove come back.
    #
    # `drop_nulls` below is the one thing that can still change the assignment: a
    # municipality with no GDP per capita cannot be stratified, so it is absent
    # from `codes` and every dyad it originates lands in `inference`.
    cities = (
        df.select(["source_code", "src_gdppc"])
          .unique("source_code")
          .drop_nulls("src_gdppc")
          .sort("source_code")
    )
    codes = cities["source_code"].to_numpy()
    income = cities["src_gdppc"].to_numpy()
    cuts = np.quantile(income, [1 / N_STRATA, 2 / N_STRATA])
    strata = np.digitize(income, cuts, right=True)

    train_codes, test_codes = train_test_split(
        codes,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=strata,
    )
    report(f"municipios   : {len(codes):,} with a GDP per capita "
           f"({df.height:,} dyads)")
    report(f"  tercile cuts: {cuts[0]:,.1f} / {cuts[1]:,.1f} pesos")
    report(f"  train       : {len(train_codes):,}")
    report(f"  test        : {len(test_codes):,}")

    # Plain lists, not pl.Series: `is_in` against a Series of the same dtype is
    # deprecated in polars >= 1.17 and warns about ambiguous semantics.
    train_cities = train_codes.tolist()
    test_cities = test_codes.tolist()

    df = df.with_columns(
        pl.col("source_code").is_in(train_cities).alias("src_train"),
        pl.col("source_code").is_in(test_cities).alias("src_test"),
        pl.col("dest_code").is_in(train_cities).alias("dst_train"),
        pl.col("dest_code").is_in(test_cities).alias("dst_test"),
    )

    report("exclusions   : none -- every municipality in the panel is retained")

    is_train = pl.col("src_train") & pl.col("dst_train")
    is_test = pl.col("src_test") & pl.col("dst_test")

    train_df = df.filter(is_train)
    test_df = df.filter(is_test)
    inference_df = df.filter(~is_train & ~is_test)

    flag_cols = ["src_train", "src_test", "dst_train", "dst_test"]
    train_df = train_df.drop(flag_cols)
    test_df = test_df.drop(flag_cols)
    inference_df = inference_df.drop(flag_cols)

    total = train_df.height + test_df.height + inference_df.height
    report("")
    report(f"train pairs     : {train_df.height:,}")
    report(f"test  pairs     : {test_df.height:,}")
    report(f"inference pairs : {inference_df.height:,}")
    report(f"partition check : {total:,} == {df.height:,}")
    if total != df.height:
        raise SystemExit("pairs do not partition cleanly")

    # A test municipality must never appear in a training dyad.
    leaked = set(train_df["source_code"]).union(train_df["dest_code"]) & set(test_codes)
    if leaked:
        raise SystemExit(f"leak: {len(leaked)} test municipios appear in train dyads")
    report("leakage check   : no test municipio appears in a train dyad")

    # --- write ---------------------------------------------------------------
    report("")
    for split_df, name in [(train_df, "train"), (test_df, "test"),
                           (inference_df, "inference")]:
        X = split_df.select(WHITELIST_FEATURES)
        y = split_df.select(TARGET)
        y_log = y.select(pl.col(TARGET).log1p().alias(f"log_{TARGET}"))

        X.write_csv(args.outdir / f"X_{name}.csv")
        y.write_csv(args.outdir / f"y_{name}.csv")
        y_log.write_csv(args.outdir / f"y_{name}_log1p.csv")
        report(f"wrote X_{name}.csv {str(X.shape):>16}  "
               f"+ y_{name}.csv, y_{name}_log1p.csv")

    # --- centroids: the models need lat/lon to build the spatial graph -------
    # Copied rather than referenced so that `data/` is a self-contained
    # modelling input directory.
    if args.centroids.exists():
        cent = pl.read_csv(args.centroids, schema_overrides={"cvegeo": pl.Utf8})
        cent = cent.with_columns(pl.col("cvegeo").str.zfill(5))
        cent.write_csv(args.outdir / "muni_centroids.csv")
        report(f"wrote muni_centroids.csv {str(cent.shape):>11}")
    else:
        report(f"WARNING centroids not found: {args.centroids}")
        report("        the GNN cannot build its spatial graph without them")

    manifest = args.outdir / "splits_manifest.md"
    manifest.write_text(
        "# modelling splits -- manifest\n\n"
        f"Generated by `models/make_splits.py` from "
        f"`{args.panel.relative_to(ROOT).as_posix()}`.\n\n"
        f"- random_state: `{args.random_state}`\n"
        f"- test_size: `{args.test_size}`\n"
        f"- strata: {N_STRATA} terciles of `src_gdppc` over source municipalities\n"
        f"- excluded municipalities: none\n\n"
        "```\n" + "\n".join(lines) + "\n```\n",
        encoding="utf-8",
    )
    print(f"\nwrote {manifest.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
