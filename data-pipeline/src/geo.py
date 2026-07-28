"""
Geometry loading and 2015->2020 boundary harmonization.

Like common.py this is a helper, not a numbered step. It exists because the
crosswalk is needed by 01_flows (to harmonize O-D codes), 02_population,
05_climate and 06_demography, and having four scripts each roll their own
version of "which municipalities changed" is how a panel ends up unbalanced.

THE PROBLEM
-----------
Mexican municipalities split. A person who reported living in municipality X in
2015 may today be described by a 2020 geography in which X has been carved into
X and Y. If the 2015-residence code is joined naively against 2020 geometry, the
split-off children appear as orphans and a silent inner join deletes them.

THE DEFAULT (config: geometry.crosswalk_strategy = aggregate_to_parent)
-----------------------------------------------------------------------
Both origin AND destination codes are mapped back to the pre-split PARENT. The
panel is then balanced -- the same municipal universe exists at both ends of the
window -- at the cost of coarser geography in a handful of places. The
alternative (`allocate`) splits the parent's flows across children by population
share, which invents within-parent variation that the data does not contain.
See README.md#boundary-harmonization.

THE SAFETY NET
--------------
`audit_codes()` does not trust the seeded crosswalk. It compares the codes
actually observed in the flows against the codes actually present in the 2020
geometry, in both directions, and reports every unmatched code. Any real
boundary change shows up there whether or not anyone remembered to add a row to
the CSV.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from common import (
    CVEGEO_DTYPE, PipelineError, ROOT, assert_cvegeo_valid, rel, resolve,
)

__all__ = [
    "load_municipios",
    "load_crosswalk",
    "seed_crosswalk_if_absent",
    "apply_crosswalk",
    "audit_codes",
    "SEED_CROSSWALK",
]


# ---------------------------------------------------------------------------
# Seed crosswalk
# ---------------------------------------------------------------------------
# Municipality creations falling inside the 2015-2020 window.
#
# WHAT "verified" MEANS HERE: the child and parent CVEGEO codes, and the names
# attached to them, were checked against INEGI's ITER 2020 municipal rows. That
# rules out the failure mode that actually bit us -- a code pointing at an
# entirely different municipality.
#
# WHAT IS STILL NOT MACHINE-VERIFIED: the parent-child RELATIONSHIP itself, i.e.
# that Coatetelco was carved specifically out of Miacatlan rather than a
# neighbour. That comes from the decrees, not from ITER, which lists
# municipalities but not their lineage. Once the Marco Geoestadistico is
# available, load_municipios() gives polygons and adjacency becomes checkable:
# a genuine parent must border its child. Until then, treat the pairing as
# well-supported but not proven.
#
# audit_codes() independently surfaces any boundary change missing from this
# list, so an omission is loud rather than silent.
_ITER_SRC = ("codes+names verified against INEGI ITER 2020 "
             "(conjunto_de_datos_iter_00CSV20.csv, checked 2026-07-21)")

SEED_CROSSWALK: list[dict[str, Any]] = [
    # --- Morelos: three municipalities created 2017 -------------------------
    # CORRECTED 2026-07-21. The first version of this table was wrong in a way
    # that would have produced silent nonsense: it paired Xoxocotla with code
    # 17020, which is actually TEPOZTLAN -- a municipality on the other side of
    # the state that has nothing to do with the split. Folding Tepoztlan's flows
    # into Xoxocotla's would have corrupted both.
    #
    # Verified child codes and names against ITER 2020, which lists Morelos with
    # 36 municipalities: 17034 Coatetelco, 17035 Xoxocotla, 17036 Hueyapan.
    dict(cvegeo_child="17034", name_child="Coatetelco",
         cvegeo_parent="17015", name_parent="Miacatlan",
         year_created=2017, verified=True,
         source_note="Morelos decree 2017; " + _ITER_SRC +
                     ". ITER: 17034=Coatetelco (pop 11,347), 17015=Miacatlan "
                     "(pop 15,802)."),
    dict(cvegeo_child="17035", name_child="Xoxocotla",
         cvegeo_parent="17017", name_parent="Puente de Ixtla",
         year_created=2017, verified=True,
         source_note="Morelos decree 2017; " + _ITER_SRC +
                     ". ITER: 17035=Xoxocotla (pop 27,805), 17017=Puente de "
                     "Ixtla (pop 40,018). NOTE 17020 is Tepoztlan, NOT Puente "
                     "de Ixtla -- an earlier version of this row had that wrong."),
    dict(cvegeo_child="17036", name_child="Hueyapan",
         cvegeo_parent="17022", name_parent="Tetela del Volcan",
         year_created=2017, verified=True,
         source_note="Morelos decree 2017; " + _ITER_SRC +
                     ". ITER: 17036=Hueyapan (pop 7,855), 17022=Tetela del "
                     "Volcan (pop 14,853). NOTE 17026 is Tlayacapan, NOT Tetela "
                     "del Volcan -- an earlier version of this row had that wrong."),

    # --- Quintana Roo: Puerto Morelos created 2016 --------------------------
    dict(cvegeo_child="23011", name_child="Puerto Morelos",
         cvegeo_parent="23005", name_parent="Benito Juarez",
         year_created=2016, verified=True,
         source_note="Quintana Roo decree 2016; " + _ITER_SRC +
                     ". ITER: 23011=Puerto Morelos, 23005=Benito Juarez."),

    # --- Baja California: San Quintin created 2020 --------------------------
    # San Felipe was in an earlier version of this table and has been REMOVED:
    # ITER 2020 lists Baja California with six municipalities (02001-02006) and
    # no San Felipe. Whatever its decree date, it is not in the 2020 census
    # universe, so no 2015-residence code can refer to it and there is nothing
    # to harmonize. A crosswalk row for a municipality that does not exist would
    # be dead weight at best and a mis-mapping risk at worst.
    dict(cvegeo_child="02006", name_child="San Quintin",
         cvegeo_parent="02001", name_parent="Ensenada",
         year_created=2020, verified=True,
         source_note="Baja California decree 2020; " + _ITER_SRC +
                     ". ITER: 02006=San Quintin, 02001=Ensenada. BC has exactly "
                     "6 municipalities in 2020; San Felipe is absent."),

    # -----------------------------------------------------------------------
    # POST-2020 creations
    # -----------------------------------------------------------------------
    # These serve a different purpose from the rows above. The 2015-2020 rows
    # harmonize FLOW codes; these harmonize GEOMETRY, so that a Marco
    # Geoestadistico edition NEWER than the census can be used. Folding a
    # post-census child back into its parent dissolves the two polygons and
    # reconstructs the pre-split boundary, which is exactly the 2020 extent.
    #
    # No flow record can ever reference these codes -- the census predates them
    # -- so including them is harmless when the geometry is 2020 and necessary
    # when it is not.
    #
    # PARENTS WERE DERIVED FROM DATA, NOT RECALLED. Method: every ITER 2020
    # locality carries coordinates and its 2020 municipal code. Point-in-polygon
    # of 189,432 localities against the post-2020 boundaries shows which 2020
    # municipality each new unit's territory came from. Five of six are
    # unambiguous at 100% of population; San Felipe is 97.1% (see its note).
    dict(cvegeo_child="02007", name_child="San Felipe",
         cvegeo_parent="02002", name_parent="Mexicali",
         year_created=2021, verified=True,
         source_note="post-2020 creation, needed only for a newer geometry "
                     "edition. Parent derived by point-in-polygon of ITER 2020 "
                     "localities: 99 localities / 19,741 people from 02002 "
                     "Mexicali (97.1%), 83 localities / 582 people from 02001 "
                     "Ensenada (2.9%). Assigned to Mexicali. CAVEAT: the "
                     "Ensenada remainder may be a genuine second donor rather "
                     "than coordinate imprecision; if so, folding San Felipe "
                     "wholly into Mexicali leaves Mexicali marginally too large "
                     "and Ensenada marginally too small."),
    dict(cvegeo_child="04013", name_child="Dzitbalche",
         cvegeo_parent="04001", name_parent="Calkini",
         year_created=2021, verified=True,
         source_note="post-2020 creation. Parent derived by point-in-polygon: "
                     "28 ITER 2020 localities / 16,573 people, 100% from 04001 "
                     "Calkini."),
    dict(cvegeo_child="12082", name_child="Las Vigas",
         cvegeo_parent="12053", name_parent="San Marcos",
         year_created=2022, verified=True,
         source_note="post-2020 creation (Guerrero). Parent derived by "
                     "point-in-polygon: 32 ITER 2020 localities / 9,942 people, "
                     "100% from 12053 San Marcos."),
    dict(cvegeo_child="12083", name_child="Nuu Savi",
         cvegeo_parent="12012", name_parent="Ayutla de los Libres",
         year_created=2022, verified=True,
         source_note="post-2020 creation (Guerrero). Parent derived by "
                     "point-in-polygon: 41 ITER 2020 localities / 11,143 people, "
                     "100% from 12012 Ayutla de los Libres."),
    dict(cvegeo_child="12084", name_child="Santa Cruz del Rincon",
         cvegeo_parent="12041", name_parent="Malinaltepec",
         year_created=2022, verified=True,
         source_note="post-2020 creation (Guerrero). Parent derived by "
                     "point-in-polygon: 16 ITER 2020 localities / 7,124 people, "
                     "100% from 12041 Malinaltepec."),
    dict(cvegeo_child="12085", name_child="San Nicolas",
         cvegeo_parent="12023", name_parent="Cuajinicuilapa",
         year_created=2022, verified=True,
         source_note="post-2020 creation (Guerrero). Parent derived by "
                     "point-in-polygon: 18 ITER 2020 localities / 7,011 people, "
                     "100% from 12023 Cuajinicuilapa."),
]

CROSSWALK_COLUMNS = [
    "cvegeo_child", "name_child", "cvegeo_parent", "name_parent",
    "year_created", "verified", "source_note",
]


def seed_crosswalk_if_absent(cfg: Mapping[str, Any],
                             logger: logging.Logger | None = None) -> Path:
    """
    Write the seed crosswalk to data/raw/ if it does not already exist.

    Lives under data/raw/ because it is an input to the pipeline, not a product
    of it -- but note it is the one file in raw/ that a human is expected to
    edit, since verifying the codes is a manual task.
    """
    path = resolve(cfg, cfg["geometry"]["crosswalk_file"])
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(SEED_CROSSWALK, columns=CROSSWALK_COLUMNS)
    df.to_csv(path, index=False, encoding="utf-8")
    if logger:
        logger.warning(
            "SEEDED crosswalk at %s with %d UNVERIFIED row(s). "
            "Verify every CVEGEO against the INEGI catalogo before trusting a build.",
            rel(path), len(df),
        )
    return path


def load_crosswalk(cfg: Mapping[str, Any],
                   logger: logging.Logger | None = None) -> pd.DataFrame:
    """Load the 2015->2020 crosswalk, seeding it first if absent."""
    path = seed_crosswalk_if_absent(cfg, logger)
    df = pd.read_csv(path, dtype={"cvegeo_child": CVEGEO_DTYPE,
                                  "cvegeo_parent": CVEGEO_DTYPE})
    missing = set(CROSSWALK_COLUMNS) - set(df.columns)
    if missing:
        raise PipelineError(f"crosswalk {path} missing columns: {sorted(missing)}")

    df["cvegeo_child"] = df["cvegeo_child"].str.zfill(5)
    df["cvegeo_parent"] = df["cvegeo_parent"].str.zfill(5)
    assert_cvegeo_valid(df["cvegeo_child"], "crosswalk.cvegeo_child")
    assert_cvegeo_valid(df["cvegeo_parent"], "crosswalk.cvegeo_parent")

    # A child that is itself a parent means a chained split. Resolve it or the
    # mapping is order-dependent.
    chained = set(df["cvegeo_child"]) & set(df["cvegeo_parent"])
    if chained:
        raise PipelineError(
            f"crosswalk has chained splits (code is both child and parent): "
            f"{sorted(chained)}. Flatten these to the ultimate parent."
        )

    if logger:
        n_unverified = int((~df["verified"].astype(bool)).sum())
        logger.info("crosswalk: %d mapping(s), %d unverified", len(df), n_unverified)
        if n_unverified:
            logger.warning(
                "%d crosswalk row(s) are UNVERIFIED. Boundary harmonization is a "
                "material analytic choice -- see README.md#boundary-harmonization.",
                n_unverified,
            )
    return df


def apply_crosswalk(codes: pd.Series, xwalk: pd.DataFrame, strategy: str,
                    logger: logging.Logger | None = None,
                    label: str = "codes") -> pd.Series:
    """
    Map post-split child codes back onto their pre-split parent.

    Returns a new Series; does not mutate. Codes with no crosswalk entry pass
    through unchanged, which is the correct behaviour for the ~2,460
    municipalities that never changed.
    """
    if strategy == "none":
        return codes
    if strategy == "allocate":
        raise NotImplementedError(
            "crosswalk_strategy='allocate' is documented in README.md but not "
            "implemented: it requires within-parent population shares and would "
            "manufacture variation the flow data does not contain. Use "
            "'aggregate_to_parent' (default) or implement allocation deliberately."
        )
    if strategy != "aggregate_to_parent":
        raise PipelineError(f"unknown crosswalk_strategy: {strategy!r}")

    mapping = dict(zip(xwalk["cvegeo_child"], xwalk["cvegeo_parent"]))
    out = codes.astype(CVEGEO_DTYPE)
    hit = out.isin(mapping.keys())
    n_hit = int(hit.sum())
    out = out.map(lambda c: mapping.get(c, c) if pd.notna(c) else c).astype(CVEGEO_DTYPE)
    if logger:
        logger.info("crosswalk applied to %s: %s value(s) remapped to parent",
                    label, f"{n_hit:,}")
    return out


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def load_municipios(cfg: Mapping[str, Any], logger: logging.Logger | None = None):
    """
    Load the INEGI Marco Geoestadistico municipal layer as a GeoDataFrame.

    Guarantees on return:
      * a `cvegeo` column, 5-char string dtype, no nulls, no duplicates
      * CRS is exactly config.crs.storage (reprojected if not)
    """
    import geopandas as gpd

    gcfg = cfg["geometry"]
    path = resolve(cfg, gcfg["municipal_layer"])
    if not path.exists():
        raise PipelineError(
            f"municipal layer not found: {rel(path)}. "
            "Run `python src/00_download.py --only marco_geoestadistico` "
            "(and fill in its URL first -- see config.yaml downloads)."
        )

    gdf = gpd.read_file(path)
    if logger:
        logger.info("READ  municipal layer  rows=%s  crs=%s", f"{len(gdf):,}", gdf.crs)

    field = gcfg["cvegeo_field"]
    if field not in gdf.columns:
        # Fall back to composing it from separate state/municipality fields,
        # which some MGN distributions ship instead of a single CVEGEO.
        from common import cvegeo_series
        ent_f = next((c for c in ("CVE_ENT", "CVEENT", "ENTIDAD") if c in gdf.columns), None)
        mun_f = next((c for c in ("CVE_MUN", "CVEMUN", "MUNICIPIO") if c in gdf.columns), None)
        if not (ent_f and mun_f):
            raise PipelineError(
                f"geometry has neither {field!r} nor a (state, municipality) pair. "
                f"Columns present: {list(gdf.columns)}"
            )
        if logger:
            logger.warning("geometry lacks %s; composing from %s + %s", field, ent_f, mun_f)
        gdf["cvegeo"] = cvegeo_series(gdf[ent_f], gdf[mun_f])
    else:
        gdf["cvegeo"] = gdf[field].astype(CVEGEO_DTYPE).str.strip().str.zfill(5)

    assert_cvegeo_valid(gdf["cvegeo"], "geometry.cvegeo")

    dup = gdf["cvegeo"].duplicated()
    if bool(dup.any()):
        # Multipart municipalities occasionally ship as separate features.
        # Dissolving is correct, but say so rather than doing it quietly.
        if logger:
            logger.warning("geometry has %d duplicate cvegeo feature(s); dissolving",
                           int(dup.sum()))
        name_f = gcfg.get("name_field")
        agg = {name_f: "first"} if name_f and name_f in gdf.columns else {}
        gdf = gdf.dissolve(by="cvegeo", aggfunc=agg).reset_index()
        gdf["cvegeo"] = gdf["cvegeo"].astype(CVEGEO_DTYPE)

    target = cfg["crs"]["storage"]
    if gdf.crs is None:
        raise PipelineError(
            "municipal layer has NO CRS. Refusing to assume one -- an unlabelled "
            "CRS is how zonal statistics end up silently offset. Set it explicitly."
        )
    if gdf.crs.to_string() != target:
        if logger:
            logger.info("reprojecting geometry %s -> %s", gdf.crs.to_string(), target)
        gdf = gdf.to_crs(target)

    keep = ["cvegeo", "geometry"]
    if gcfg.get("name_field") in gdf.columns:
        keep.insert(1, gcfg["name_field"])
    return gdf[keep].rename(columns={gcfg.get("name_field", ""): "nomgeo"})


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def assert_geometry_vintage(cfg: Mapping[str, Any], gdf,
                            logger: logging.Logger | None = None) -> None:
    """
    Check the geometry edition against the census municipal universe.

    INEGI publishes a Marco Geoestadistico every year or two, and the download
    page offers several. Taking the wrong edition fails in two ways:

      LOUD  -- municipalities created after the census (Campeche added
               Seybaplaya and Dzitbalche in 2021, Baja California's San Felipe
               post-dates the 2020 count) appear in the geometry with no
               possible flow. The panel spine is built from the geometry
               universe, so these become rows for units that did not exist
               during the migration window.

      QUIET -- and far worse. When a municipality splits, the PARENT polygon
               shrinks. Population-weighted centroids, zonal climate and zonal
               GDP would then be computed over the newer territory and joined to
               census-vintage flows and population. Every number stays plausible
               while describing slightly the wrong ground.

    The census tabulado gives the authoritative municipal universe for the
    round, so comparing against it catches the mismatch before any raster pass.
    Runs only if interim/population.parquet exists (i.e. after 02_population).
    """
    from common import read_interim, PipelineError as _PE  # local: avoid cycle

    try:
        pop = read_interim(cfg, "population")
    except Exception:  # noqa: BLE001 -- population not built yet; nothing to check
        if logger:
            logger.info("geometry vintage check skipped (run 02_population first)")
        return

    census = set(pop["cvegeo"].dropna().astype(str))
    year = cfg["project"]["census_year"]

    # Compare the HARMONIZED universe. A newer geometry edition is legitimate as
    # long as the crosswalk folds its post-census creations back into their
    # parents -- that dissolve reconstructs the census-vintage boundaries. So
    # the question is not "is this the 2020 file" but "after harmonization, does
    # this describe the 2020 municipal universe".
    raw_geom = set(gdf["cvegeo"].dropna().astype(str))
    xwalk = load_crosswalk(cfg, logger)
    harmonized = apply_crosswalk(
        pd.Series(sorted(raw_geom), dtype=CVEGEO_DTYPE), xwalk,
        cfg["geometry"]["crosswalk_strategy"], None, "geometry vintage check")
    geom = set(harmonized.dropna().astype(str))

    extra = sorted(geom - census)
    missing = sorted(census - geom)

    if logger:
        n_folded = len(raw_geom) - len(geom)
        logger.info("geometry vintage check: raw=%s  after crosswalk=%s  "
                    "census %s=%s", f"{len(raw_geom):,}", f"{len(geom):,}",
                    year, f"{len(census):,}")
        if n_folded:
            logger.info("  %d post-census municipality/ies folded into their "
                        "parents by the crosswalk", n_folded)

    if not extra and not missing:
        if logger:
            logger.info("  PASS  geometry matches the %s census universe exactly", year)
        return

    # Name the offending units where we can -- a bare code is not enough to work
    # out which parent it split from.
    names = {}
    if "nomgeo" in getattr(gdf, "columns", []):
        names = dict(zip(gdf["cvegeo"].astype(str), gdf["nomgeo"].astype(str)))

    def _fmt(codes):
        return [f"{c} ({names[c]})" if c in names else c for c in codes[:15]]

    detail = []
    if extra:
        detail.append(f"{len(extra)} in the geometry but NOT in the {year} census:")
        detail += [f"    {s}" for s in _fmt(extra)]
    if missing:
        detail.append(f"{len(missing)} in the {year} census but NOT in the geometry:")
        detail += [f"    {s}" for s in _fmt(missing)]

    # A newer edition is USABLE, because folding a child back into its parent
    # unions their polygons and restores the pre-split extent exactly. Emit a
    # ready-to-edit crosswalk stub rather than just refusing.
    stub = ""
    if extra and not missing:
        rows = "\n".join(
            f"{c},{names.get(c, '<name>')},<PARENT_CVEGEO>,<parent name>,"
            f"<year>,False,\"created after {year}; folded back to its {year} parent\""
            for c in extra[:15]
        )
        stub = (
            "\nTHIS IS RECOVERABLE. Every extra code is a municipality created after\n"
            f"the {year} census. Folding each back into its parent DISSOLVES the two\n"
            "polygons together, which reconstructs the pre-split boundary exactly --\n"
            "so a newer edition gives the correct geometry once the crosswalk knows\n"
            "about the newer splits.\n\n"
            "Append rows like these to\n"
            f"  {cfg['geometry']['crosswalk_file']}\n"
            "filling in each parent (the municipality it was carved out of):\n\n"
            + rows + "\n\n"
            "Then re-run. The flows themselves are unaffected: they are coded to the\n"
            f"{year} universe and can never reference a code created after it.\n"
        )

    raise _PE(
        "GEOMETRY VINTAGE MISMATCH\n\n  " + "\n  ".join(detail) + "\n\n"
        f"The municipal universe in the geometry does not match the {year} census.\n"
        "The usual cause is a Marco Geoestadistico edition from a different year --\n"
        "INEGI offers several and only the census-matched one lines up directly.\n\n"
        f"Cleanest fix: use the edition labelled 'Censo de Poblacion y Vivienda\n"
        f"{year}' (library record upc=889463807469 for the 2020 round).\n"
        + stub +
        "\nWhy this is a hard stop rather than a warning: when a municipality splits,\n"
        "the PARENT polygon shrinks. Centroids, zonal climate and zonal GDP computed\n"
        "on the wrong vintage stay entirely plausible while describing the wrong\n"
        "ground, and nothing downstream would reveal it."
    )


def audit_codes(observed_codes: pd.Series, geometry_codes: pd.Series,
                label: str, logger: logging.Logger | None = None) -> pd.DataFrame:
    """
    Compare codes observed in the data against codes present in 2020 geometry.

    This is the safety net that does not depend on anyone having remembered to
    add a crosswalk row. Returns a frame of unmatched codes in BOTH directions:

      orphan_in_data     : appears in the flows, absent from 2020 geometry.
                           Usually a pre-split code, or a bad sentinel.
      absent_from_data   : exists in 2020 geometry, never appears in the data.
                           Usually a newly created municipality (fine), or a
                           genuine zero-flow municipality (also fine), or a
                           parsing bug (not fine).

    Reported, never silently dropped.
    """
    obs = set(pd.Series(observed_codes).dropna().astype(str).unique())
    geo = set(pd.Series(geometry_codes).dropna().astype(str).unique())

    orphans = sorted(obs - geo)
    absentees = sorted(geo - obs)

    rows = [{"code": c, "issue": "orphan_in_data", "context": label} for c in orphans]
    rows += [{"code": c, "issue": "absent_from_data", "context": label} for c in absentees]
    out = pd.DataFrame(rows, columns=["code", "issue", "context"])

    if logger:
        logger.info("AUDIT %-22s observed=%s  geometry=%s  orphans=%d  absent=%d",
                    label, f"{len(obs):,}", f"{len(geo):,}", len(orphans), len(absentees))
        if orphans:
            logger.warning("  %d orphan code(s) in %s (in data, not in 2020 geometry): %s",
                           len(orphans), label, orphans[:20])
            logger.warning("  -> these need a crosswalk row, or are unhandled sentinels.")
        if absentees:
            logger.info("  %d code(s) in geometry with no %s record: %s",
                        len(absentees), label, absentees[:20])
    return out


