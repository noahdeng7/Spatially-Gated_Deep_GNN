"""Choropleth maps of Brazil for three per-municipality features.

  python data/plot_muni_maps.py            # all three + the combined panel

Renders, on the 2010 IBGE municipal boundaries (5,565 munis) the pipeline is
pinned to:

  urban_area_pct  MapBiomas col. 10.1, 2006-2010 mean  (feature_land_use.parquet)
  mean_income     censobr 2010, V0010-weighted          (census_income.parquet)
  temp            ERA5-Land via GEE, 2005-2010 mean °F  (feature_climate_baseline.parquet)

Darker = more of the quantity in every map.

CLASSIFICATION — quantiles, not equal intervals
-----------------------------------------------
urban_area_pct spans 0 → 99.8% but its median is 0.38%: on an equal-interval
ramp ~99% of Brazil collapses into the single lightest class and the map shows
nothing. All three maps therefore use 7 quantile classes (equal muni counts per
class) with the real value range printed on every legend swatch, so the reader
sees both the rank structure and the magnitudes. Income and temperature would
survive equal intervals; they use quantiles too so the three maps are read the
same way.

COLOR
-----
One hue per map, light→dark (sequential encoding). The blue ramp is the
documented sequential ramp; the orange and aqua ramps are derived here in OKLCH
from the palette's slot-2 / slot-3 hues, reusing the blue ramp's exact lightness
profile — so equal darkness means equal rank across all three maps, and each
ramp stays monotone in L and single-hue.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent          # <repo>/data
PIPELINE = HERE.parent / "Data_Pipeline"
INTERMEDIATE = PIPELINE / "data" / "intermediate"
SHAPEFILE = PIPELINE / "data" / "raw" / "shapefile" / "BR_Municipios_2010.shp"
OUT_DIR = HERE / "maps"

EQUAL_AREA = "EPSG:5880"   # SIRGAS 2000 / Brazil Polyconic — same CRS the pipeline uses

# Chart chrome (light mode) — palette.md § Chart chrome & ink
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
NO_DATA = "#e1e0d9"        # gridline step, reads as "absent" not as a low value
STATE_LINE = "#c3c2b7"

N_CLASSES = 7

# BR_Municipios_2010.shp carries 5,567 polygons: the 5,565 municipalities plus two
# pseudo-muni codes IBGE uses for the big Rio Grande do Sul coastal lagoons. They
# have no census records, and a land-cover or air-temperature class shaded over open
# water would read as a municipal value — so they are drawn as water and held out of
# the quantile classification.
WATER_CODES = {"4300001": "Lagoa Mirim", "4300002": "Lagoa dos Patos"}


# ---------------------------------------------------------------------------
# OKLab / OKLCH (Ottosson) — used only to derive the two non-documented ramps.
# ---------------------------------------------------------------------------
def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def hex_to_oklch(hex_color: str) -> tuple[float, float, float]:
    rgb = np.array([int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    lms = _M1 @ _srgb_to_linear(rgb)
    L, a, b = _M2 @ np.cbrt(lms)
    return float(L), float(np.hypot(a, b)), float(np.arctan2(b, a))


def oklch_to_rgb(L: float, C: float, h: float) -> np.ndarray:
    lab = np.array([L, C * np.cos(h), C * np.sin(h)])
    lms = np.linalg.solve(_M2, lab) ** 3
    return _linear_to_srgb(np.linalg.solve(_M1, lms))


def _in_gamut(rgb: np.ndarray) -> bool:
    return bool(np.all(rgb >= -1e-4) and np.all(rgb <= 1 + 1e-4))


def _max_chroma(L: float, h: float) -> float:
    """Largest in-sRGB-gamut chroma for this lightness + hue (bisection)."""
    lo, hi = 0.0, 0.4
    for _ in range(40):
        mid = (lo + hi) / 2
        if _in_gamut(oklch_to_rgb(L, mid, h)):
            lo = mid
        else:
            hi = mid
    return lo


def _rgb_to_hex(rgb: np.ndarray) -> str:
    v = np.clip(np.round(rgb * 255), 0, 255).astype(int)
    return "#{:02x}{:02x}{:02x}".format(*v)


# Documented sequential ramp (palette.md § Sequential hue), steps 100..700.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def derive_ramp(base_hex: str) -> list[str]:
    """A 7-step light→dark ramp on `base_hex`'s hue, matching BLUE_RAMP's L profile.

    Lightness is copied step-for-step from the documented blue ramp (so darkness
    means the same rank in every map) and chroma is the blue step's chroma,
    clamped to what this hue can actually hold at that lightness.
    """
    _, _, h = hex_to_oklch(base_hex)
    out = []
    for step in BLUE_RAMP:
        L, C, _ = hex_to_oklch(step)
        out.append(_rgb_to_hex(oklch_to_rgb(L, min(C, 0.95 * _max_chroma(L, h)), h)))
    return out


ORANGE_RAMP = derive_ramp("#eb6834")   # palette slot 2
AQUA_RAMP = derive_ramp("#1baf7a")     # palette slot 3


def assert_monotone(ramp: list[str], name: str) -> None:
    """Sequential ramps are checked for lightness monotonicity, not adjacency CVD."""
    Ls = [hex_to_oklch(c)[0] for c in ramp]
    dL = np.diff(Ls)
    assert np.all(dL < 0), f"{name} ramp is not monotone in L: {np.round(Ls, 3)}"
    assert np.all(np.abs(dL) >= 0.06), f"{name} adjacent ΔL < 0.06: {np.round(np.abs(dL), 3)}"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _code(s: pd.Series) -> pd.Series:
    """IBGE 7-digit municipality code as a zero-padded string."""
    return s.astype(str).str.extract(r"(\d+)", expand=False).str.zfill(7).str[:7]


def load_munis() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(SHAPEFILE).to_crs(EQUAL_AREA)
    key = "code_muni" if "code_muni" in gdf.columns else "CD_MUN"
    gdf["code_muni"] = _code(gdf[key])
    return gdf[["code_muni", "geometry"]].drop_duplicates("code_muni")


def load_series(parquet: str, column: str) -> pd.Series:
    df = pd.read_parquet(INTERMEDIATE / parquet)
    df["code_muni"] = _code(df["code_muni"])
    return df.set_index("code_muni")[column]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def classify(values: pd.Series, n: int = N_CLASSES) -> np.ndarray:
    """Quantile class edges (n+1 of them), de-duplicated if the tail is degenerate."""
    edges = np.nanquantile(values.dropna().to_numpy(), np.linspace(0, 1, n + 1))
    edges = np.unique(edges)
    edges[0] -= 1e-9      # make the lowest edge inclusive for BoundaryNorm
    return edges


def draw_map(ax, gdf: gpd.GeoDataFrame, col: str, ramp: list[str],
             fmt, title: str, subtitle: str) -> None:
    edges = classify(gdf[col])
    cmap = ListedColormap(ramp[:len(edges) - 1])
    norm = BoundaryNorm(edges, cmap.N)

    has = gdf[col].notna()
    gdf.loc[~has].plot(ax=ax, color=NO_DATA, edgecolor="none")
    gdf.loc[has].plot(ax=ax, column=col, cmap=cmap, norm=norm, edgecolor="none")

    # State outlines for orientation — recessive, above the fills.
    states = gdf.assign(uf=gdf["code_muni"].str[:2]).dissolve("uf")
    states.boundary.plot(ax=ax, color=STATE_LINE, linewidth=0.4)

    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=15, color=INK_PRIMARY,
                 fontweight="600", pad=14)
    ax.text(0, 1.005, subtitle, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9.5, color=INK_SECONDARY)

    # Legend: one swatch per class, labelled with its real value range, plus
    # no-data when any muni is missing. labelspacing gives the 2px-gap feel.
    handles = [Patch(facecolor=c, edgecolor="none",
                     label=f"{fmt(edges[i] if i else edges[0] + 1e-9)} – {fmt(edges[i + 1])}")
               for i, c in enumerate(cmap.colors)][::-1]   # darkest first
    if (~has).any():
        handles.append(Patch(facecolor=NO_DATA, edgecolor="none",
                             label="Coastal lagoon"))
    leg = ax.legend(handles=handles, loc="lower left", frameon=False,
                    fontsize=8.5, labelspacing=0.55, handlelength=1.1,
                    handleheight=1.1, borderpad=0, handletextpad=0.6)
    for text in leg.get_texts():
        text.set_color(INK_SECONDARY)


SPECS = [
    dict(key="urban", parquet="feature_land_use.parquet", column="urban_area_pct",
         ramp=BLUE_RAMP,
         title="Urban land cover",
         subtitle="% of municipal area classified urban · MapBiomas col. 10.1, 2006–2010 mean",
         fmt=lambda v: f"{v:,.2f}%" if v < 10 else f"{v:,.1f}%"),
    dict(key="income", parquet="census_income.parquet", column="mean_income",
         ramp=AQUA_RAMP,
         title="Mean monthly income",
         subtitle="BRL per income-earning person · Census 2010 (censobr), V0010-weighted",
         # "$" must be escaped or matplotlib parses the label as mathtext.
         fmt=lambda v: f"R\\${v:,.0f}"),
    dict(key="temp", parquet="feature_climate_baseline.parquet", column="temp",
         ramp=ORANGE_RAMP,
         title="Mean annual temperature",
         subtitle="°F, annual mean of daily-mean 2 m temperature · ERA5-Land, 2005–2010 mean",
         fmt=lambda v: f"{v:.1f}°"),
]

def source_note(n_munis: int) -> str:
    return (f"2010 IBGE municipal boundaries ({n_munis:,} munis) · {N_CLASSES} quantile "
            "classes (equal municipality counts) · SIRGAS 2000 / Brazil Polyconic")


def main() -> None:
    for ramp, name in ((BLUE_RAMP, "blue"), (ORANGE_RAMP, "orange"), (AQUA_RAMP, "aqua")):
        assert_monotone(ramp, name)
    print(f"  ramps OK  orange={ORANGE_RAMP}\n            aqua={AQUA_RAMP}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })

    munis = load_munis()
    print(f"  {len(munis)} municipality polygons", flush=True)

    frames = []
    for spec in SPECS:
        s = load_series(spec["parquet"], spec["column"])
        g = munis.merge(s.rename(spec["column"]), left_on="code_muni",
                        right_index=True, how="left")
        g.loc[g["code_muni"].isin(WATER_CODES), spec["column"]] = np.nan
        frames.append(g)
        n_missing = int(g[spec["column"]].isna().sum())
        print(f"  {spec['column']:<16} matched {len(g) - n_missing}/{len(g)}", flush=True)

        fig, ax = plt.subplots(figsize=(8.2, 8.8))
        draw_map(ax, g, spec["column"], spec["ramp"], spec["fmt"],
                 spec["title"], spec["subtitle"])
        fig.text(0.02, 0.018, source_note(len(munis) - len(WATER_CODES)), fontsize=7.5, color=INK_MUTED)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        out = OUT_DIR / f"map_{spec['key']}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"  wrote {out}", flush=True)

    # Combined 3-panel figure (small multiples — one ramp each, own legend).
    fig, axes = plt.subplots(1, 3, figsize=(19, 7.6))
    for ax, spec, g in zip(axes, SPECS, frames):
        draw_map(ax, g, spec["column"], spec["ramp"], spec["fmt"],
                 spec["title"], spec["subtitle"])
    fig.suptitle("Brazil by municipality — darker is more", x=0.012, y=0.99,
                 ha="left", fontsize=18, color=INK_PRIMARY, fontweight="600")
    fig.text(0.012, 0.02, source_note(len(munis) - len(WATER_CODES)), fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    out = OUT_DIR / "map_panel.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
