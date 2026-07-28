"""
fetch_ce2019_saic.py -- fetch Censos Economicos 2019 municipal VACB from the
SAIC JSON API.

docs/MANUAL_DOWNLOADS.md section 4 says SAIC is "an interactive query tool ...
there is no stable direct URL". That is true of the *download links* but not of
the backend: the SAIC single-page app talks to a JSON API at

    https://www.inegi.org.mx/app/api/saic/

which this script drives directly (reverse-engineered from saicRedisign.min.js,
2026-07-21). Three facts discovered on the way, all of which matter downstream:

1. SAIC labels the census by REFERENCE year, not collection year: the CE 2019
   data are requested as anios=[2018]. This settles README TODO #7 -- the
   figures describe fiscal 2018, so gdp.ppp.deflator_from_year_to_base must be
   the 2018->2017 deflator (0.950530), not the 2019->2017 one.

2. The VACB variable is A131A, "Valor agregado censal bruto (millones de
   pesos)" -- MILLIONS, not the "miles de pesos" the config default assumed.
   gdp.censos_economicos.units must be millions_of_pesos.

3. Years must be sent as JSON integers; the same query with "2018" as a string
   returns an empty "no information" response rather than an error.

Output: data/raw/censos_economicos/ce2019_municipal.csv with columns
    CVEGEO, NOM_ENT, NOM_MUN, ANIO, VA_BRUTO
one row per municipality (all-sector municipal total, "Con totales"), VA_BRUTO
in millones de pesos, exactly what 04_gdp.py reads via gdp.censos_economicos.

Municipalities SAIC returns no row for are municipalities with no (publishable)
economic-census activity; they are listed at the end of the run and left absent
rather than zero-filled -- 04_gdp.py's LEFT join reports them as missing, which
is the honest representation of "no census establishments", and the coverage is
asserted to be >= 95% of the geography catalog so a silent API regression
cannot masquerade as rural emptiness.

Run:
    python src/fetch_ce2019_saic.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    PipelineError, ensure_dirs, get_logger, load_config, rel, resolve, run_step,
)

STEP = "fetch_ce2019_saic"

API = "https://www.inegi.org.mx/app/api/saic"
# SAIC census id 6 = the current SAIC application (serves 2003..2023).
CENSUS_APP_ID = 6
# CE 2019 reports fiscal 2018 and SAIC indexes it that way.
REFERENCE_YEAR = 2018
VACB_CODE = "A131A"

TIMEOUT = 120
CHUNK = 100          # municipalities per POST; well under any server cap
MIN_COVERAGE = 0.95  # fraction of catalog municipalities that must return data


def _get(session: requests.Session, path: str) -> dict:
    r = session.get(f"{API}/{path}", timeout=TIMEOUT)
    r.raise_for_status()
    d = r.json()
    if not d.get("success"):
        raise PipelineError(f"SAIC GET {path} -> success=false: {d}")
    return d


def _municipal_catalog(session, log) -> dict[str, tuple[str, str]]:
    """CVEGEO -> (state name, municipality name) from the SAIC geography tree."""
    states = _get(session, f"ageos/seg/00/1/{CENSUS_APP_ID}/")["list"]
    if len(states) != 32:
        raise PipelineError(f"SAIC geography catalog lists {len(states)} states, "
                            "expected 32 -- refusing to continue on a partial tree")
    out: dict[str, tuple[str, str]] = {}
    for st in states:
        munis = _get(session, f"ageos/seg/{st['key']}/1/{CENSUS_APP_ID}/")["list"]
        for m in munis:
            key = str(m["key"]).strip().zfill(5)
            out[key] = (st["name"], m["name"])
        log.info("  %s %-25s %4d municipios", st["key"], st["name"], len(munis))
        time.sleep(0.1)   # be polite; the whole catalog is 33 requests
    return out


def _query_vacb(session, cvegeos: list[str], log) -> dict[str, float]:
    """POST consulta/tabla for A131A over the given municipalities."""
    payload = {
        "anios": [REFERENCE_YEAR],          # int, NOT str -- see module docstring
        "ageos": cvegeos,
        "actecos": [],                      # empty + total=True -> all-sector total
        "varcens": [{"nom": VACB_CODE, "pos": 0}],
        "indicators": [],
        "stratums": [0],
        "calcs": [],
        "total": True,
        "orden": "1",
        "desc": False,
        "page": 0,
        "reg": len(cvegeos),
    }
    r = session.post(f"{API}/consulta/tabla/{CENSUS_APP_ID}/",
                     json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    d = r.json()
    if not d.get("success"):
        # "No exite información disponible" is what an all-empty chunk returns;
        # legitimate for a chunk of tiny municipalities with no establishments.
        log.warning("    chunk of %d returned no data: %s",
                    len(cvegeos), d.get("msg"))
        return {}

    rows = json.loads(d["data"])["info"]
    out: dict[str, float] = {}
    for row in rows:
        # "enti": "01 Aguascalientes", "muni": "001 Aguascalientes"
        ent = str(row["enti"]).split()[0]
        mun = str(row["muni"]).split()[0]
        cvegeo = (ent + mun).zfill(5)
        (vacb_entry,) = row["zxc"]          # exactly one requested variable
        (label, value), = vacb_entry.items()
        if VACB_CODE not in label:
            raise PipelineError(f"asked for {VACB_CODE}, got {label!r}")
        if value is not None:
            out[cvegeo] = float(value)
    return out


def main() -> int:
    cfg = load_config()
    ensure_dirs(cfg)
    log = get_logger(STEP, cfg)
    log.info("=" * 78)
    log.info("FETCH Censos Economicos 2019 municipal VACB via the SAIC JSON API")
    log.info("=" * 78)
    log.info("reference year sent to the API: %d (CE 2019 reports fiscal 2018)",
             REFERENCE_YEAR)

    session = requests.Session()
    session.headers["User-Agent"] = "mx-migration pipeline (research use)"

    log.info("building municipal catalog from the SAIC geography tree ...")
    catalog = _municipal_catalog(session, log)
    log.info("catalog: %s municipios", f"{len(catalog):,}")

    keys = sorted(catalog)
    values: dict[str, float] = {}
    for i in range(0, len(keys), CHUNK):
        chunk = keys[i:i + CHUNK]
        values.update(_query_vacb(session, chunk, log))
        log.info("  %5d / %d queried, %5d with data",
                 min(i + CHUNK, len(keys)), len(keys), len(values))
        time.sleep(0.2)

    coverage = len(values) / len(catalog)
    log.info("VACB rows: %s of %s municipios (%.1f%%)",
             f"{len(values):,}", f"{len(catalog):,}", 100 * coverage)
    if coverage < MIN_COVERAGE:
        raise PipelineError(
            f"only {coverage:.1%} of municipalities returned a VACB value "
            f"(threshold {MIN_COVERAGE:.0%}). That pattern looks like an API "
            "change or a half-failed run, not like genuinely absent data -- "
            "refusing to write a file that would silently understate the country."
        )

    missing = sorted(set(catalog) - set(values))
    if missing:
        log.warning("%d municipios have no VACB row (left absent, not zero-filled):",
                    len(missing))
        for k in missing[:20]:
            log.warning("    %s %s, %s", k, catalog[k][1], catalog[k][0])
        if len(missing) > 20:
            log.warning("    ... and %d more", len(missing) - 20)

    out_path = resolve(cfg, cfg["gdp"]["censos_economicos"]["file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig: read_csv_smart tries that first, and Excel opens it correctly.
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["CVEGEO", "NOM_ENT", "NOM_MUN", "ANIO", "VA_BRUTO"])
        for k in sorted(values):
            w.writerow([k, catalog[k][0], catalog[k][1], REFERENCE_YEAR,
                        repr(values[k])])

    total = sum(values.values())
    log.info("WROTE %s  rows=%s  total VACB = %s millones de pesos",
             rel(out_path), f"{len(values):,}", f"{total:,.0f}")
    log.info("")
    log.info("REMINDERS (both already reflected in config.yaml if this ran "
             "from the committed tree):")
    log.info("  gdp.censos_economicos.units: millions_of_pesos  (A131A is "
             "millones)")
    log.info("  gdp.ppp.deflator_from_year_to_base: 0.950530  (2018 -> 2017)")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_step(main, STEP))
