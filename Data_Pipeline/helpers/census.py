"""Censobr-2010-derived per-municipality features (trimmed set).

The 8 functions cover education, infrastructure, age, sex, dwelling, household,
income, and labour aggregations. Each returns a DataFrame keyed by `code_muni`
and, unless noted, weighted by the census expansion factor `V0010`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import (
    read_household,
    read_population,
    weighted_pct,
)

# ---------------------------------------------------------------------------
# Education (V6400, persons aged >= 25)
# ---------------------------------------------------------------------------
def education() -> pd.DataFrame:
    df = read_population(["V6400", "V6036"])
    df["age"] = pd.to_numeric(df["V6036"], errors="coerce")
    df["edu"] = pd.to_numeric(df["V6400"], errors="coerce")
    df["w"]   = pd.to_numeric(df["V0010"], errors="coerce")
    df = df[(df["age"] >= 25) & df["edu"].notna() & (df["edu"] != 5)]
    df["primary_or_less"] = df["edu"].isin([1, 2]).astype(float)
    df["secondary"]       = (df["edu"] == 3).astype(float)
    df["tertiary"]        = (df["edu"] == 4).astype(float)

    g = df.groupby("code_muni").apply(lambda x: pd.Series({
        "primary_or_less": float((x["w"] * x["primary_or_less"]).sum()),
        "secondary":       float((x["w"] * x["secondary"]).sum()),
        "tertiary":        float((x["w"] * x["tertiary"]).sum()),
    })).reset_index()
    total = g["primary_or_less"] + g["secondary"] + g["tertiary"]
    g["primary_or_less_%_2010"] = 100.0 * g["primary_or_less"] / total
    g["secondary_%_2010"]       = 100.0 * g["secondary"]       / total
    g["tertiary_%_2010"]        = 100.0 * g["tertiary"]        / total
    return g[["code_muni", "primary_or_less_%_2010",
              "secondary_%_2010", "tertiary_%_2010"]]


# ---------------------------------------------------------------------------
# Infrastructure (household parquet)
# ---------------------------------------------------------------------------
def infrastructure() -> pd.DataFrame:
    df = read_household(["V0207", "V0209", "V0211"])
    df["w"] = pd.to_numeric(df["V0010"], errors="coerce")
    for c in ("V0207", "V0209", "V0211"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    def agg(x):
        ev = x[x["V0211"].notna()]
        wv = x[x["V0209"].notna()]
        sv = x[x["V0207"].notna()]
        return pd.Series({
            "population_electricity": float(ev["w"].sum()),
            "no_electricity":         float((ev["w"] * (ev["V0211"] == 3)).sum()),
            "population_piped_water": float(wv["w"].sum()),
            "no_piped_water":         float((wv["w"] * (wv["V0209"] == 3)).sum()),
            "population_sewage":      float(sv["w"].sum()),
            "no_sewage":              float((sv["w"] * (~sv["V0207"].isin([1, 2]))).sum()),
        })

    g = df.groupby("code_muni").apply(agg).reset_index()
    g["no_electricity_%_2010"] = 100.0 * g["no_electricity"] / g["population_electricity"]
    g["no_piped_water_%_2010"] = 100.0 * g["no_piped_water"] / g["population_piped_water"]
    g["no_sewage_%_2010"]      = 100.0 * g["no_sewage"]      / g["population_sewage"]
    return g[["code_muni", "no_electricity_%_2010",
              "no_piped_water_%_2010", "no_sewage_%_2010"]]


# ---------------------------------------------------------------------------
# Age structure
#
# Per presentation slide 1: pct_18_35 uses V6036.  Denominator is restricted to
#     rows with a valid age (NaN-age rows dropped) — explicit user choice on
#     top of the slide.
# ---------------------------------------------------------------------------
def age() -> pd.DataFrame:
    df = read_population(["V6036"])
    df["age"] = pd.to_numeric(df["V6036"], errors="coerce")     # slide 1: V6036
    df["w"]   = pd.to_numeric(df["V0010"], errors="coerce")
    df.loc[(df["age"] < 0) | (df["age"] > 120), "age"] = np.nan

    df["a_18_35"] = ((df["age"] >= 18) & (df["age"] <= 35)).astype(float)

    def agg(x):
        # Drop NaN-age rows from the denominator — slide 1 formula computed
        # over valid-age population only.
        v = x.dropna(subset=["age"])
        wv = float(v["w"].sum())

        # Slide 1: pct_18_35 = 100 × Σ(w · age_18_35) / Σ w   (valid-age only)
        pct18 = 100.0 * float((v["w"] * v["a_18_35"]).sum()) / wv \
                  if wv > 0 else np.nan
        return pd.Series({"pct_18_35_2010": pct18})

    return df.groupby("code_muni").apply(agg).reset_index()


# ---------------------------------------------------------------------------
# Sex
# ---------------------------------------------------------------------------
def sex() -> pd.DataFrame:
    df = read_population(["V0601"])
    df["w"] = pd.to_numeric(df["V0010"], errors="coerce")
    df["V0601"] = pd.to_numeric(df["V0601"], errors="coerce")
    df = df[df["V0601"].isin([1, 2])]
    df["male"] = (df["V0601"] == 1).astype(float)
    return df.groupby("code_muni").apply(lambda x: pd.Series({
        "male_pct_w_2010": weighted_pct(x["male"], x["w"]),
    })).reset_index()


# ---------------------------------------------------------------------------
# Urban / rural
# ---------------------------------------------------------------------------
def dwelling() -> pd.DataFrame:
    df = read_population(["V1006"])
    df["w"] = pd.to_numeric(df["V0010"], errors="coerce")
    df["V1006"] = pd.to_numeric(df["V1006"], errors="coerce")
    df = df[df["V1006"].isin([1, 2])]
    df["urban"] = (df["V1006"] == 1).astype(float)
    # urban stored as a fraction (0-1), not percent — it's the mean of the 0/1
    # indicator; multiply by 100 for a percentage.
    out = df.groupby("code_muni").apply(lambda x: pd.Series({
        "pct_urban_w_2010": weighted_pct(x["urban"], x["w"]) / 100.0,
    })).reset_index()
    # Imputation step: weighted_pct's `100 * num/den` then `/100` can drift
    # above 1.0 by ~1e-9 on munis that are 100% urban because V0010 sums in
    # float64 accumulate roundoff (~5 munis affected, all "true 100% urban").
    # Clip to the mathematical bounds so downstream consumers can treat the
    # column as a true fraction in [0, 1].
    out["pct_urban_w_2010"] = out["pct_urban_w_2010"].clip(lower=0.0, upper=1.0)
    return out


# ---------------------------------------------------------------------------
# Household size (slide 3A: V0401 from household parquet),
# dependency ratio (slide 4: HOUSEHOLD-based), total population
# ---------------------------------------------------------------------------
def household() -> pd.DataFrame:
    # ---- Household parquet for size (V0401) per slide 3A ------------------
    hdf = read_household(["V0401"])
    hdf["w"]     = pd.to_numeric(hdf["V0010"], errors="coerce")
    hdf["V0401"] = pd.to_numeric(hdf["V0401"], errors="coerce")

    def _size_agg(x):
        wsum = float(x["w"].sum())
        return pd.Series({
            # Slide 3A: Mean household size = Σ(V0010 × V0401) / Σ V0010
            "hh_mean_size_2010":
                (x["w"] * x["V0401"]).sum() / wsum if wsum > 0 else np.nan,
        })
    sizes = hdf.groupby("code_muni").apply(_size_agg).reset_index()

    # ---- Person parquet for dependency ratio and total population ---------
    pdf = read_population(["V0300", "V6036"])
    pdf["w"]   = pd.to_numeric(pdf["V0010"], errors="coerce")
    pdf["age"] = pd.to_numeric(pdf["V6036"], errors="coerce")

    # ---- Slide 4: Household-based dependency ratio -----------------------
    #
    # Procedure:
    #   1) Classify persons:  dependents = ages 0-14 OR 65+,  working = 15-64
    #   2) Per household h:    DR_h = Σ(w · dep) / Σ(w · wrk)
    #   3) Per muni:           DR_m = Σ(w_h · DR_h) / Σ w_h  × 100
    # Households with zero working-age members produce undefined DR_h — they
    # are excluded from the muni-level weighted mean (their weight too).
    pdf["is_dep"] = ((pdf["age"] <= 14) | (pdf["age"] >= 65)).astype(float)
    pdf["is_wrk"] = ((pdf["age"] >= 15) & (pdf["age"] <= 64)).astype(float)

    # Weighted in-household counts using person weights.
    pdf["wd"] = pdf["w"] * pdf["is_dep"]
    pdf["ww"] = pdf["w"] * pdf["is_wrk"]

    hh = pdf.groupby(["code_muni", "V0300"]).agg(
        dep=("wd", "sum"),
        wrk=("ww", "sum"),
        w_h=("w", "first"),       # household weight = first member's V0010
    ).reset_index()

    # Drop households with no working-age members (DR_h undefined).
    valid_hh = hh[hh["wrk"] > 0].copy()
    valid_hh["dr_total"] = valid_hh["dep"] / valid_hh["wrk"]

    def _muni_dr(x):
        w_sum = float(x["w_h"].sum())
        if w_sum == 0:
            return pd.Series({"dependency_ratio_2010": np.nan})
        return pd.Series({
            "dependency_ratio_2010":
                100.0 * float((x["w_h"] * x["dr_total"]).sum()) / w_sum,
        })
    dr = valid_hh.groupby("code_muni").apply(_muni_dr).reset_index()

    # total_pop: weighted sum of all persons in the muni (expanded population).
    pop = pdf.groupby("code_muni")["w"].sum().rename("total_pop_2010").reset_index()

    return (sizes
              .merge(dr,  on="code_muni", how="outer")
              .merge(pop, on="code_muni", how="outer"))


# ---------------------------------------------------------------------------
# Income transfers + earnings
# Vectorized groupby sums instead of groupby.apply(lambda) so peak memory
# stays under ~1 GB (apply on 20M rows can spike to 3+ GB and OOM-kill).
# ---------------------------------------------------------------------------
def _weighted_pct_by(values: pd.Series, weights: pd.Series,
                     mask: pd.Series, key: pd.Series) -> pd.Series:
    """Per-key weighted % = 100 × Σ(w·v)/Σ(w), all vectorized."""
    v = (values * weights).where(mask)
    w = weights.where(mask)
    num = v.groupby(key).sum()
    den = w.groupby(key).sum()
    return (100.0 * num / den.replace(0, np.nan))


def income() -> pd.DataFrame:
    df = read_population(["V0656", "V0657", "V6527"])
    df["w"]  = pd.to_numeric(df["V0010"], errors="coerce")
    df["p"]  = pd.to_numeric(df["V0656"], errors="coerce")
    df["bf"] = pd.to_numeric(df["V0657"], errors="coerce")
    df["inc"] = pd.to_numeric(df["V6527"], errors="coerce")

    code = df["code_muni"]
    pct_p  = _weighted_pct_by((df["p"]  == 1).astype(float), df["w"], df["p"].isin([0, 1]),  code)
    pct_bf = _weighted_pct_by((df["bf"] == 1).astype(float), df["w"], df["bf"].isin([0, 1]), code)
    pct_p.name, pct_bf.name = "pct_pension_2010", "pct_bolsa_familia_2010"

    # Slide 7: mean income is weighted by V0010.
    inc_mask = df["inc"].notna() & (df["inc"] >= 0)
    num = (df["inc"] * df["w"]).where(inc_mask).groupby(code).sum()
    den = df["w"].where(inc_mask).groupby(code).sum().replace(0, np.nan)
    mean_inc = (num / den).rename("mean_income_brl_2010")

    return pd.concat([pct_p, pct_bf, mean_inc], axis=1).reset_index()


# ---------------------------------------------------------------------------
# Labour: unemployment % and informal %  (vectorized, low-memory)
# ---------------------------------------------------------------------------
def labour() -> pd.DataFrame:
    df = read_population(["V6900", "V6920", "V6930"])
    df["w"] = pd.to_numeric(df["V0010"], errors="coerce")
    for c in ("V6900", "V6920", "V6930"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    in_lf    = (df["V6900"] == 1).astype(float)
    unemp    = ((df["V6900"] == 1) & (df["V6920"] == 2)).astype(float)
    emp      = (df["V6920"] == 1).astype(float)
    informal = (emp.astype(bool) & df["V6930"].isin([3, 6])).astype(float)

    code = df["code_muni"]
    lf_g  = (df["w"] * in_lf).groupby(code).sum()
    un_g  = (df["w"] * unemp).groupby(code).sum()
    em_g  = (df["w"] * emp).groupby(code).sum()
    inf_g = (df["w"] * informal).groupby(code).sum()

    out = pd.DataFrame({
        "unemp_pct_2010":    100.0 * un_g  / lf_g.replace(0, np.nan),
        "informal_pct_2010": 100.0 * inf_g / em_g.replace(0, np.nan),
    }).reset_index()
    return out


ALL_CENSUS = {
    "education":      education,
    "infrastructure": infrastructure,
    "age":            age,
    "sex":            sex,
    "dwelling":       dwelling,
    "household":      household,
    "income":         income,
    "labour":         labour,
}
