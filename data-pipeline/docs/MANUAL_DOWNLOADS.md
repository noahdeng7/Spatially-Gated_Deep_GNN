# Manual downloads — RESOLVED 2026-07-21

All four inputs that required a human are now on disk. This file records **how
each one was actually obtained**, because three of the four routes were not the
documented ones, and each will matter again the day the data needs re-fetching.

Check state at any time with:

```bash
python src/00_download.py --verify
```

> ### Still true, still the most important thing on this page
>
> **INEGI serves missing files as HTTP 200 with an HTML error page**, not a
> 404. A wrong URL returns a ~1.4–2.3 KB page saying *"Esta liga ya no existe"*
> with a success status. Check the `Content-Type`, not the status code.

---

## 1. Census microdata — RESOLVED (with a finding that changed a default)

**What was learned:** the full-count **Cuestionario Básico microdata is not
public.** INEGI's microdatos page describes its CB files as examples that
"no permiten hacer ningún tipo de inferencia", offered so users can test
syntax before submitting jobs to the *Laboratorio de microdatos* / remote
processing. No amount of URL-guessing would ever have found a public CB file,
because none exists.

**What was fetched instead:** the public instrument, the Cuestionario Ampliado
~10% sample, national scope, one 486 MB zip, live as of 2026-07-21:

```
https://www.inegi.org.mx/contenidos/programas/ccpv/2020/microdatos/Censo2020_CA_eum_csv.zip
```

Naming decodes as: `CA` = ampliado, `eum` = Estados Unidos Mexicanos
(national). The earlier dead guess `Censo2020_CA_nal_csv.zip` failed because
the scope token is `eum`, not `nal`. Per-state files
(`Censo2020_CA_ags_csv.zip`, …) are also live for most states, but several
states use non-obvious abbreviations (camp/coah/chis/chih/tamps/tlax all 404
under those spellings), so the national file is the sane choice.

The zip contains `Personas00.CSV`, `Viviendas00.CSV`, `Migrantes00.CSV`. The
person table's header was inspected directly: `FACTOR`, `ENT`, `MUN`, `EDAD`,
`ENT_PAIS_RES_5A`, `MUN_RES_5A` all present.

**Placement quirk:** `01_flows.py` classifies instruments by the `_CA`
filename marker, and INEGI's national files don't carry it. The extracted
tables are therefore renamed:

```
Personas00.CSV  ->  data/raw/censo2020/Personas_CA_00.csv   (and siblings)
```

The zip itself is kept unmodified next to them as the raw record.

**Config consequence:** `flows.use_basic_questionnaire: false`. If you obtain
the real CB through the Microdata Lab, place its person tables (no `_CA` in
the names) under `data/raw/censo2020/` and flip the flag back.

---

## 2. Marco Geoestadístico — RESOLVED (fallback source)

The official 2020 portal download remains dead / JS-locked. In use instead:
INEGI's ArcGIS Feature Service, fetched by `src/fetch_mgn_arcgis.py` into
`data/raw/mgn2022/00mun.gpkg` (2,475 municipios, WGS84).

Two caveats, both documented in that script's docstring: the service is
labelled *Prueba* (test), and it is the **December 2022** vintage — 2,475
municipios vs the census's 2,469. The crosswalk folds the six post-census
creations back into their parents, and `geo.assert_geometry_vintage()`
verifies the harmonized universe matches 2020. Prefer the official product if
INEGI ever republishes a working link (`upc=889463807469`).

---

## 3. CONAPO municipal projections — RESOLVED (Wayback Machine)

Every live route is dead: `conapo.gob.mx` 404s (a real 404), the
datos.gob.mx CKAN API was retired in the portal revamp, and the new
`repodatos.atdt.gob.mx` mirror carries only the **state-level 2023-base**
series — which cannot substitute for a municipal panel.

The official files were recovered **byte-identical from the Internet
Archive** (RAR v4 archives, content timestamps Aug 2019):

```
https://web.archive.org/web/20220621144457id_/http://www.conapo.gob.mx/work/models/CONAPO/Datos_Abiertos/Proyecciones2018/base_municipios_final_datos_01.rar
https://web.archive.org/web/20220306151221id_/http://www.conapo.gob.mx/work/models/CONAPO/Datos_Abiertos/Proyecciones2018/base_municipios_final_datos_02.rar
```

(The `id_` suffix asks Wayback for the original bytes, no banner injection.)

Extracted and concatenated (byte-level, second header dropped) into the path
config expects:

```
data/raw/conapo/proyecciones_municipales.csv
```

Latin-1, columns `RENGLON, CLAVE, CLAVE_ENT, NOM_ENT, MUN, SEXO, AÑO,
EDAD_QUIN, POB` — long panel, municipality × year × sex × 5-year age band.
Verified: 32 states, **2,457 municipios** (the 2015 Intercensal universe, as
CONAPO documents), 2015 population total **121,347,800**. `02_population.py`
recognises `CLAVE` as the key and `POB` as the population column and collapses
the sex/age dimensions.

---

## 4. Censos Económicos 2019 — RESOLVED (SAIC has an API after all)

The "interactive tool with no public API" turns out to run on a JSON API at
`https://www.inegi.org.mx/app/api/saic/`. `src/fetch_ce2019_saic.py` drives it
directly — geography catalog first, then chunked VACB queries — and writes:

```
data/raw/censos_economicos/ce2019_municipal.csv
  CVEGEO, NOM_ENT, NOM_MUN, ANIO, VA_BRUTO
```

Result: **2,463 of 2,478** municipios with data (99.4%). The 15 absent are
post-CE2019 municipal creations (San Quintín, Seybaplaya, Dzitbalché, Ñuu
Savi, …) plus a couple of confidentiality suppressions; they are left absent,
not zero-filled. National total **9,983,798 millones de pesos**, matching the
published CE 2019 figure.

Three API facts that are now load-bearing config:

1. **Reference year.** SAIC indexes the quinquennial censuses by reference
   year — CE 2019 answers to `anios=[2018]`. So the extract reports fiscal
   2018 and `gdp.ppp.deflator_from_year_to_base` is **0.950530** (2018→2017),
   resolving README TODO #7.
2. **Units.** The VACB variable is `A131A`, "Valor agregado censal bruto
   (**millones** de pesos)". `gdp.censos_economicos.units` is
   `millions_of_pesos` — the old `thousands_of_pesos` guess would have been a
   1000× level error that the correlation gate could never catch.
3. **Types matter.** Years must be JSON integers; `"2018"` as a string returns
   an empty "no information" response, not an error.

---

## Order of attack (historical)

The original priority table is preserved for context; every row is done.

| Priority | Input | Unblocks | Outcome |
|---|---|---|---|
| 1 | Census microdata | `01_flows` and everything keyed on it | CA national zip, live URL |
| 2 | Marco Geoestadístico | `03_distance`, `05_climate`, `04_gdp` geometry | ArcGIS fallback, Dec-2022 vintage |
| 3 | Censos Económicos | primary GDP + correlation cross-check | SAIC JSON API |
| 4 | CONAPO | 2015 origin population | Wayback Machine recovery |
