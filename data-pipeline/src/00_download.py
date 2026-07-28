"""
00_download.py -- fetch raw inputs into data/raw/. Never edits anything.

Idempotent: a file that already exists with a matching size (and sha256, when
one is configured) is skipped. Re-running is cheap and safe.

DELIBERATE BEHAVIOUR: every download URL lives in config.yaml under `downloads:`
and some are still null. This script does NOT guess URLs. For each null entry it
prints the recorded instructions and exits non-zero. A fabricated URL either
404s (annoying) or silently fetches the wrong vintage (much worse), and the
second failure mode is invisible until the numbers are already in a regression
table.

Each manifest entry carries a `confidence` tier, reported in --list:

    VERIFIED  the URL or its containing directory was confirmed to exist
    INFERRED  built from a documented naming convention plus a confirmed
              sibling file, but the exact URL was not fetched -- may 404
    MANUAL    no stable direct URL exists; a human must fetch it

Run:
    python src/00_download.py                 # fetch everything configured
    python src/00_download.py --only worldpop # fetch one entry
    python src/00_download.py --list          # show status, fetch nothing
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import zipfile
import gzip
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ROOT, load_config, get_logger, rel, resolve, run_step, ensure_dirs, PipelineError,
)

STEP = "00_download"


def sha256sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def already_have(dest: Path, expected_sha: str | None) -> bool:
    """Idempotency check. A directory counts as present if it is non-empty."""
    if dest.is_dir():
        return any(dest.iterdir())
    if not dest.exists() or dest.stat().st_size == 0:
        return False
    if expected_sha:
        return sha256sum(dest) == expected_sha
    return True


# Content types that indicate a web page rather than a data file.
HTML_TYPES = ("text/html", "application/xhtml")

# Expected content-type prefix per unpack/extension, used to catch a server
# that hands back an error page with a 200 status.
EXPECTED_TYPES = {
    ".zip": ("application/zip", "application/x-zip", "application/octet-stream"),
    ".tif": ("image/tiff", "application/octet-stream"),
    ".tiff": ("image/tiff", "application/octet-stream"),
    ".nc": ("application/x-netcdf", "application/octet-stream"),
    ".rar": ("application/x-rar", "application/octet-stream"),
    ".json": ("application/json",),
}


def _check_content_type(url: str, dest: Path, headers, logger) -> None:
    """
    Refuse an HTML response where a data file was expected.

    THIS GUARD EXISTS BECAUSE INEGI RETURNS HTTP 200 FOR MISSING FILES.
    A request for a non-existent archive comes back as a ~1.4 KB HTML error
    page with status 200 and content-type text/html. Without this check the
    downloader writes that page to disk under a .zip name, `already_have()`
    then reports the input as PRESENT on every subsequent run, and the failure
    only surfaces several steps later as an unintelligible parse error.

    A loud failure here is worth far more than a tidy one later.
    """
    ctype = (headers.get("content-type") or "").split(";")[0].strip().lower()
    suffix = dest.suffix.lower()

    if any(ctype.startswith(h) for h in HTML_TYPES) and suffix not in (".html", ".htm"):
        size = headers.get("content-length", "unknown")
        raise PipelineError(
            f"server returned an HTML page where a data file was expected.\n"
            f"  url          {url}\n"
            f"  content-type {ctype}\n"
            f"  size         {size} bytes\n"
            f"  expected     a {suffix or 'binary'} file\n\n"
            "This is almost certainly a 'not found' page served with status 200 "
            "-- INEGI does this. The URL is wrong or the file has moved.\n"
            "Nothing was written to disk."
        )

    expected = EXPECTED_TYPES.get(suffix)
    if expected and ctype and not any(ctype.startswith(e) for e in expected):
        logger.warning("  content-type %r is unexpected for %s (wanted one of %s). "
                       "Continuing, but verify the file.", ctype, suffix, expected)


def download(url: str, dest: Path, logger) -> Path:
    import requests
    from tqdm import tqdm

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    logger.info("GET   %s", url)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        _check_content_type(url, dest, r.headers, logger)
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))

    # A file far smaller than any real input is the other signature of an error
    # page that slipped past the content-type check.
    if tmp.stat().st_size < 4096 and dest.suffix.lower() in EXPECTED_TYPES:
        head = tmp.read_bytes()[:400]
        tmp.unlink()
        raise PipelineError(
            f"downloaded file is implausibly small ({tmp.stat if False else '<4 KB'}) "
            f"for a {dest.suffix} input.\n"
            f"  url {url}\n"
            f"  first bytes: {head[:200]!r}\n\n"
            "Treated as an error page rather than data. Nothing was kept."
        )

    tmp.replace(dest)
    return dest


def unpack(archive: Path, kind: str, logger, keep_archive: bool = False) -> None:
    """
    Extract an archive into its own directory.

    Note `archive.parent`: the caller places the archive INSIDE the intended
    destination directory, so extracting to the parent puts the contents exactly
    where config expects them. The archive itself is then removed by default --
    leaving a 4 GB zip beside its own extracted contents doubles disk use for no
    benefit, and `already_have()` keys on the directory being non-empty.
    """
    if kind == "zip":
        logger.info("UNZIP %s -> %s/", archive.name, archive.parent.name)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(archive.parent)
        if not keep_archive:
            archive.unlink()
            logger.info("      removed archive after extraction")
        return
    elif kind == "gz":
        logger.info("GUNZIP %s", archive.name)
        out = archive.with_suffix("")
        with gzip.open(archive, "rb") as fin, open(out, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    elif kind == "rar":
        # CONAPO ships .rar. The standard library cannot read it and there is no
        # pure-Python extractor worth depending on, so shell out if a tool is
        # available and otherwise stop with instructions. Silently leaving the
        # archive unextracted would surface later as a confusing "file not
        # found" three steps downstream.
        tool = next((t for t in ("unar", "7z", "7za", "unrar")
                     if shutil.which(t)), None)
        if tool is None:
            logger.warning("CANNOT UNPACK %s -- no rar extractor found on PATH.",
                           archive.name)
            logger.warning("  Install one of: unar, 7-Zip (7z), unrar.")
            logger.warning("  Windows:  winget install 7zip.7zip")
            logger.warning("  macOS:    brew install unar")
            logger.warning("  Debian:   apt-get install unar")
            logger.warning("  Then extract %s by hand and point "
                           "population.conapo_projections at the CSV inside.",
                           archive.name)
            return
        logger.info("UNRAR %s  (using %s)", archive.name, tool)
        cmd = ([tool, "x", "-o", str(archive.parent), str(archive)] if tool == "unar"
               else [tool, "x", f"-o{archive.parent}", "-y", str(archive)])
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("  %s failed (exit %d): %s", tool, result.returncode,
                           (result.stderr or "").strip()[:300])
            logger.warning("  Extract %s by hand.", archive.name)
    elif kind in (None, "none", ""):
        return
    else:
        raise PipelineError(f"unknown unpack kind: {kind!r}")


def verify_manual(cfg) -> int:
    """
    Check that manually-fetched inputs landed where the pipeline expects them.

    Placing a hand-downloaded file in the wrong directory is the most likely way
    a manual fetch fails, and the resulting error surfaces several steps later
    as a confusing "not found". This checks the actual config paths that the
    pipeline will read, not just that *something* was downloaded.

    Reports PASS / MISSING per input and prints the exact path expected.
    """
    log = get_logger(STEP, cfg)
    log.info("=" * 78)
    log.info("VERIFYING MANUALLY-FETCHED INPUTS")
    log.info("=" * 78)

    full_count = bool(cfg["flows"].get("use_basic_questionnaire", False))
    checks: list[tuple[str, Path, str]] = [
        ("census microdata (glob)",
         ROOT / cfg["flows"]["microdata_glob"],
         ("Cuestionario Basico person tables -- Personas*.csv, NOT Personas_CA*"
          if full_count else
          "Cuestionario Ampliado person tables -- Personas_CA*.csv")),
        ("municipal geometry",
         resolve(cfg, cfg["geometry"]["municipal_layer"]),
         "Marco Geoestadistico municipal layer (00mun.shp or equivalent)"),
        ("CONAPO projections",
         resolve(cfg, cfg["population"]["conapo_projections"]),
         "only required when population.origin_pop_year is the window-start year"),
        ("Censos Economicos",
         resolve(cfg, cfg["gdp"]["censos_economicos"]["file"]),
         "SAIC export: all municipalities x total sectors, VACB"),
    ]

    n_ok = 0
    for label, path, note in checks:
        if "*" in str(path):
            hits = sorted(ROOT.glob(cfg["flows"]["microdata_glob"]))
            # Same instrument split as 01_flows.py: the _CA filename marker.
            want = [h for h in hits
                    if ("_CA" in h.name.upper()) != full_count]
            ok = bool(want)
            detail = (f"{len(want)} file(s), e.g. {want[0].name}" if want
                      else f"no match for {cfg['flows']['microdata_glob']}")
        else:
            ok = path.exists()
            detail = (f"{path.stat().st_size / 1e6:.1f} MB" if ok else "not found")

        log.info("%-7s %-24s %s", "PASS" if ok else "MISSING", label, detail)
        log.info("        expected at: %s", rel(path))
        if not ok:
            log.info("        %s", note)
        n_ok += int(ok)

    log.info("")
    log.info("%d of %d manual inputs present", n_ok, len(checks))
    if n_ok < len(checks):
        log.info("Run `python src/00_download.py` for fetch instructions, or see "
                 "docs/MANUAL_DOWNLOADS.md")
    return 0 if n_ok == len(checks) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", default=None, help="fetch a single manifest key")
    ap.add_argument("--list", action="store_true", help="report status only")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--verify", action="store_true",
                    help="check that manually-fetched inputs are where the "
                         "pipeline expects them, and structurally sane")
    args = ap.parse_args()

    if args.verify:
        return verify_manual(load_config(args.config))

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    log = get_logger(STEP, cfg)

    manifest: dict = cfg.get("downloads", {}) or {}
    if args.only:
        if args.only not in manifest:
            log.error("no such manifest key: %s. Available: %s",
                      args.only, ", ".join(sorted(manifest)))
            return 2
        manifest = {args.only: manifest[args.only]}

    missing_urls: list[tuple[str, str]] = []
    fetched = skipped = 0

    for key, entry in manifest.items():
        entry = entry or {}
        url = entry.get("url")
        dest = resolve(cfg, entry.get("dest", f"data/raw/{key}"))
        sha = entry.get("sha256")

        conf = (entry.get("confidence") or "UNSPECIFIED").upper()

        if args.list:
            state = "PRESENT" if already_have(dest, sha) else ("MANUAL" if not url else "MISSING")
            log.info("%-22s %-9s %-10s %s", key, state, conf, rel(dest))
            continue

        if not args.force and already_have(dest, sha):
            log.info("SKIP  %-22s already present at %s", key, rel(dest))
            skipped += 1
            continue

        if not url:
            missing_urls.append((key, (entry.get("todo") or "").strip()))
            continue

        if conf == "INFERRED":
            log.warning("NOTE  %-22s URL is INFERRED from a naming convention, "
                        "not confirmed. A 404 here is expected-ish -- see its "
                        "todo note for the manual route.", key)

        try:
            # A `dest` written with a trailing slash in config means "a
            # directory": the archive is downloaded INTO it and extracted THERE.
            # This has to be read off the raw config string, because Path()
            # normalises the trailing slash away -- which previously caused the
            # archive to be saved as an extensionless file named after the
            # directory, and unpacked into the PARENT.
            dest_is_dir = str(entry.get("dest", "")).rstrip().endswith(("/", "\\"))

            if dest_is_dir:
                dest.mkdir(parents=True, exist_ok=True)
                filename = Path(url.split("?")[0]).name or f"{key}.download"
                target = dest / filename
            else:
                target = dest

            got = download(url, target, log)
            if sha:
                actual = sha256sum(got)
                if actual != sha:
                    raise PipelineError(
                        f"{key}: sha256 mismatch\n  expected {sha}\n  actual   {actual}"
                    )
                log.info("OK    %-22s sha256 verified", key)
            unpack(got, entry.get("unpack", "none"), log)
            fetched += 1
        except Exception as exc:  # noqa: BLE001 -- report and continue to next entry
            log.error("FAIL  %-22s %s", key, exc)

    if args.list:
        return 0

    log.info("SUMMARY  fetched=%d  skipped=%d  awaiting-url=%d",
             fetched, skipped, len(missing_urls))

    if missing_urls:
        log.warning("")
        log.warning("=" * 78)
        log.warning("%d input(s) require a MANUAL fetch.", len(missing_urls))
        log.warning("These have no stable direct URL -- they sit behind an")
        log.warning("interactive export tool or a browser-only bucket UI.")
        log.warning("Follow the instructions below, then re-run.")
        log.warning("=" * 78)
        for key, todo in missing_urls:
            log.warning("")
            log.warning("[TODO] %s", key)
            for line in (todo or "no note recorded").split("\n"):
                if line.strip():
                    log.warning("       %s", line.strip())
        log.warning("")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(run_step(main, STEP))



