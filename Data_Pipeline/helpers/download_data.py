"""Stage raw inputs the pipeline needs into ./raw_data/, reproducibly.

  python download_data.py                  # stage everything (auto sources)
  python download_data.py --only mapbiomas # one group
  python download_data.py --verify         # re-check pinned files' SHA-256, no download

Retrieval is driven by data_manifest.MANIFEST — the single, committed source of
truth for URLs, versions, destinations, and SHA-256 checksums. Every fetch is
idempotent (skip if present and, for pinned entries, checksum-valid) and pinned
entries are verified after download, so a silent upstream change fails loudly
instead of poisoning the dataset.

What this stages automatically (no auth needed):
  - censobr 2010 parquets            (GitHub release, pinned tag)
  - IBGE municipality shapefile+gpkg (geobr package, pinned year)
  - IBGE crop CSVs 2005-2010         (SIDRA HTML export)
  - MapBiomas Coverage 10.1 xlsx     (Google Drive, pinned id + SHA-256)
  - INPE fire focos 2006-2010        (per-year zips → CSV)
  - HOTOSM roads/rail/water          (HDX API-resolved; version floats)

What still needs one-time manual setup (see show_manual_setup):
  - Google Earth Engine auth (climate + NDVI) — or ship intermediate/*.parquet
  - Vulnerability polygons — geobr normally auto-fetches; manual fallback paths

The pipeline reads from raw_data/ unconditionally; this script just populates it.
Anything missing is reported as SKIP by pipeline.py.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

# Feature loaders (biomas_loader, transport_loader) are imported lazily below to
# reuse their download helpers; put ../feature_loaders on sys.path so that works
# whether this is run standalone or via run_pipeline.py.
_FL = Path(__file__).resolve().parent.parent / "feature_loaders"
if str(_FL) not in sys.path:
    sys.path.insert(0, str(_FL))

from utils import RAW
from data_manifest import MANIFEST, GROUPS, FIRE_YEARS, by_group, pinned, Dataset

# Windows consoles default to cp1252; force UTF-8 so the arrows/ellipsis in the
# log lines don't crash a native (non-Docker) run. No-op where stdout is already
# UTF-8 (Linux, Delta).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _present_and_valid(dest: Path, sha256: str | None) -> bool:
    """True if `dest` already exists non-empty and (if pinned) matches its hash."""
    if not (dest.exists() and dest.stat().st_size > 0):
        return False
    if sha256:
        got = _sha256(dest)
        if got.lower() != sha256.lower():
            print(f"  STALE {dest.name}: sha256 {got[:12]}… != pinned "
                  f"{sha256[:12]}… — re-downloading")
            return False
    return True


def _verify_after(dest: Path, sha256: str | None) -> None:
    if not sha256:
        return
    got = _sha256(dest)
    if got.lower() != sha256.lower():
        raise ValueError(
            f"SHA-256 mismatch for {dest.name}: expected {sha256}, got {got}. "
            "Upstream may have changed — update data_manifest if the new file is correct."
        )
    print(f"        sha256 OK ({got[:12]}…)")


def _download_http(url: str, dest: Path, sha256: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if _present_and_valid(dest, sha256):
        print(f"  SKIP {dest.relative_to(RAW)}  ({_human(dest.stat().st_size)} cached)")
        return
    print(f"  GET  {dest.relative_to(RAW)}  ← {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(dest)
    print(f"        done ({_human(dest.stat().st_size)})")
    _verify_after(dest, sha256)


def _download_gdrive(file_id: str, dest: Path, sha256: str | None = None) -> None:
    """Download a Google Drive file, handling the large-file confirm-token flow.

    Drive serves files >~25 MB behind a virus-scan interstitial. The modern
    drive.usercontent.google.com endpoint honours `confirm=t` in one shot; if
    Google still returns the HTML page we parse its form fields and retry.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if _present_and_valid(dest, sha256):
        print(f"  SKIP {dest.relative_to(RAW)}  ({_human(dest.stat().st_size)} cached)")
        return
    import requests
    print(f"  GET  {dest.relative_to(RAW)}  ← Google Drive id {file_id}")
    endpoint = "https://drive.usercontent.google.com/download"
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.Session() as s:
        params = {"id": file_id, "export": "download", "confirm": "t"}
        r = s.get(endpoint, params=params, stream=True, timeout=300)
        r.raise_for_status()
        if "text/html" in r.headers.get("Content-Type", ""):
            # Fallback: pull confirm/uuid tokens out of the interstitial form.
            import re
            fields = dict(re.findall(r'name="([^"]+)"\s+value="([^"]*)"', r.text))
            fields.setdefault("id", file_id)
            fields["export"] = "download"
            r = s.get(endpoint, params=fields, stream=True, timeout=300)
            r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    tmp.rename(dest)
    print(f"        done ({_human(dest.stat().st_size)})")
    _verify_after(dest, sha256)


def _stage(ds: Dataset) -> None:
    """Dispatch one manifest entry by method (http / gdrive)."""
    dest = RAW / ds.dest
    try:
        if ds.method == "http":
            _download_http(ds.url, dest, ds.sha256)
        elif ds.method == "gdrive":
            _download_gdrive(ds.gdrive_id, dest, ds.sha256)
        else:
            raise ValueError(f"_stage cannot handle method={ds.method!r}")
    except Exception as e:
        print(f"  FAIL {ds.dest}: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Per-group stagers
# ---------------------------------------------------------------------------
def download_censobr() -> None:
    print("\n[censobr] 2010 census parquets (~1 GB total)")
    for ds in by_group("censobr"):
        _stage(ds)


def download_shapefile() -> None:
    print("\n[geobr] Municipality shapefile + GeoPackage")
    try:
        import geobr
    except ImportError:
        print("  geobr not installed; add it to the environment (requirements.txt).")
        return
    for ds in by_group("shapefile"):
        dest = RAW / ds.dest
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  SKIP {ds.dest}")
            continue
        try:
            print(f"  GET  {ds.dest}  ← geobr.read_municipality(year={ds.geobr['year']})")
            dest.parent.mkdir(parents=True, exist_ok=True)
            geobr.read_municipality(year=ds.geobr["year"]).to_file(
                dest, driver=ds.geobr["driver"])
            print(f"        done ({_human(dest.stat().st_size)})")
        except Exception as e:
            print(f"  FAIL {ds.dest}: {e}")


def download_ibge_crops() -> None:
    print("\n[ibge_crops] State-level crop pivot tables (SIDRA CSV)")
    for ds in by_group("ibge_crops"):
        _stage(ds)


def download_mapbiomas() -> None:
    print("\n[mapbiomas] Coverage 10.1 xlsx (Google Drive, pinned id + SHA-256)")
    for ds in by_group("mapbiomas"):
        _stage(ds)


def download_fire() -> None:
    print(f"\n[fire] INPE BDQueimadas focos {FIRE_YEARS[0]}-{FIRE_YEARS[-1]} "
          "(one zip per year → CSV)")
    try:
        from biomas_loader import _download_inpe_focos
    except Exception as e:
        print(f"  FAIL biomas_loader import: {e}")
        return
    for y in FIRE_YEARS:
        try:
            csv_path = _download_inpe_focos(y)
            print(f"  OK   {csv_path.relative_to(RAW)}  ({_human(csv_path.stat().st_size)})")
        except Exception as e:
            print(f"  FAIL focos {y}: {e}")


def download_hotosm() -> None:
    print("\n[hotosm] Brazil roads / railways / waterways (HDX API-resolved)")
    try:
        from transport_loader import _download as ft_download, LAYERS
    except Exception as e:
        print(f"  FAIL transport_loader import: {e}")
        return
    for dataset, _col in LAYERS:
        try:
            path = ft_download(dataset)
            print(f"  OK   {path.relative_to(RAW)}  ({_human(path.stat().st_size)})")
        except Exception as e:
            print(f"  FAIL {dataset}: {e}")


def show_manual_setup() -> None:
    print("""
[manual] One-time auth + manual steps (not stageable unattended)
  Google Earth Engine (ERA5 climate + MODIS NDVI):
      Sign up at https://signup.earthengine.google.com
      Host: `earthengine authenticate`, then
            `gcloud auth application-default login`.
      Or skip GEE entirely by shipping the cached intermediate/*.parquet
      (climate + ndvi feature checkpoints) alongside the repo.

  Vulnerability polygons (only if geobr's IPEA mirror is unreachable):
      geobr normally pulls these automatically. Manual fallback paths:
      IBGE BATER:  https://www.ibge.gov.br/geociencias/informacoes-ambientais/
                   estudos-ambientais/21538-populacao-em-areas-de-risco-no-brasil.html
                   → raw_data/vulnerability/bater_areas_risco.shp
      FUNAI TIs:   https://www.gov.br/funai/pt-br/atuacao/terras-indigenas/
                   geoprocessamento-e-mapas/geprocessamento
                   → raw_data/vulnerability/funai_terras_indigenas.shp
      INSA Semiárido (2017): https://portal.insa.gov.br/sgvm
                   → raw_data/vulnerability/insa_semiarido.shp
""")


# ---------------------------------------------------------------------------
# Verify-only mode
# ---------------------------------------------------------------------------
def verify_pinned() -> int:
    """Re-check every pinned file's SHA-256 without downloading. Returns exit code."""
    print("\n[verify] SHA-256 of pinned datasets")
    bad = 0
    for ds in pinned():
        dest = RAW / ds.dest
        if not (dest.exists() and dest.stat().st_size > 0):
            print(f"  MISSING {ds.dest}")
            bad += 1
            continue
        got = _sha256(dest)
        if got.lower() == ds.sha256.lower():
            print(f"  OK      {ds.dest}  ({got[:12]}…)")
        else:
            print(f"  MISMATCH {ds.dest}\n           expected {ds.sha256}\n           got      {got}")
            bad += 1
    print(f"\n  {len(pinned()) - bad}/{len(pinned())} pinned files verified.")
    return 1 if bad else 0


# ---------------------------------------------------------------------------
_FUNCS = {
    "censobr":    download_censobr,
    "shapefile":  download_shapefile,
    "ibge_crops": download_ibge_crops,
    "mapbiomas":  download_mapbiomas,
    "fire":       download_fire,
    "hotosm":     download_hotosm,
    "manual":     show_manual_setup,
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", choices=list(_FUNCS), default=None,
                   help="Run only one group (default: all).")
    p.add_argument("--verify", action="store_true",
                   help="Re-check pinned files' SHA-256 and exit (no download).")
    args = p.parse_args()

    print("=" * 64)
    print("  Raw data stager → ./raw_data/   (manifest-driven, checksummed)")
    print("=" * 64)

    if args.verify:
        return verify_pinned()

    todo = list(_FUNCS) if args.only is None else [args.only]
    for s in todo:
        _FUNCS[s]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
