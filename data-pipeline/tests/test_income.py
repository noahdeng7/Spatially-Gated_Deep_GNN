"""
Unit tests for 08_income.py -- the INEGI ICMM 2020 household-income covariate.

Two things matter here and both are tested against synthetic fixtures (never the
real download):

  1. The long -> wide pivot. The file is one row per (ent, mun, est); est selects
     value / se / ci_low / ci_high / cv. State and national aggregates (mun==000)
     must be dropped, and every municipality must come out with a point estimate.

  2. The split-child fold. Household income is INTENSIVE, so a child folded into
     its parent must be POPULATION-WEIGHTED averaged, not summed. This is the one
     place income differs from population/value-added, and getting it wrong
     (summing, or an unweighted mean) is silent, so it is pinned explicitly.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from conftest import load_step

LOG = logging.getLogger("test_income")
LOG.addHandler(logging.NullHandler())


@pytest.fixture(scope="module")
def income_mod():
    return load_step("08_income")


def _write_icmm(path, rows):
    """rows: list of (ent, mun, {est: value}). Writes the long ICMM layout."""
    recs = []
    for ent, mun, ests in rows:
        for est, val in ests.items():
            recs.append({"ent": ent, "mun": mun, "est": est, "icpth": val})
    pd.DataFrame(recs, columns=["ent", "mun", "est", "icpth"]).to_csv(
        path, index=False, encoding="utf-8")


def _full(value, se=10.0, lo=None, hi=None, cv=1.0):
    """All five estimators for one municipality."""
    return {"1": value, "2": se,
            "3": lo if lo is not None else value - 50,
            "4": hi if hi is not None else value + 50, "5": cv}


def _income_config(tmp_path, weight="population"):
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    return {
        "project": {"census_year": 2020, "migration_window_years": 5},
        "paths": {"raw": str(raw), "interim": str(tmp_path / "interim"),
                  "processed": str(tmp_path / "processed"),
                  "logs": str(tmp_path / "logs"), "reports": str(tmp_path / "reports")},
        "geometry": {"crosswalk_strategy": "aggregate_to_parent",
                     "crosswalk_file": str(raw / "crosswalk.csv")},
        "population": {"census_2020_tabulado": str(raw / "iter.csv")},
        "income": {"file": str(raw / "icmm.csv"), "harmonize_weight": weight,
                   "cv_caution_threshold": 15, "cv_unreliable_threshold": 30},
    }


def _write_crosswalk(path, pairs):
    """pairs: list of (child, parent)."""
    pd.DataFrame([
        dict(cvegeo_child=c, name_child="Child", cvegeo_parent=p,
             name_parent="Parent", year_created=2017, verified=True,
             source_note="synthetic test fixture ITER 2020")
        for c, p in pairs
    ]).to_csv(path, index=False)


def _write_iter(path, pops):
    """pops: dict cvegeo -> POBTOT. Writes LOC==0000 municipal rows plus noise."""
    recs = []
    for cvegeo, pop in pops.items():
        recs.append({"ENTIDAD": cvegeo[:2], "MUN": cvegeo[2:], "LOC": "0000",
                     "NOM_MUN": "M", "POBTOT": pop})
        # a locality row that must be filtered out (LOC != 0000)
        recs.append({"ENTIDAD": cvegeo[:2], "MUN": cvegeo[2:], "LOC": "0001",
                     "NOM_MUN": "M", "POBTOT": pop // 2})
    pd.DataFrame(recs).to_csv(path, index=False)


class TestLoadIcmm:
    def test_drops_aggregates_and_pivots(self, income_mod, tmp_path):
        cfg = _income_config(tmp_path)
        _write_icmm(cfg["income"]["file"], [
            ("0", "0", _full(50000)),       # national aggregate  -> dropped
            ("1", "0", _full(58000)),       # state aggregate     -> dropped
            ("1", "1", _full(1000, se=10, lo=950, hi=1050, cv=2.0)),
            ("1", "2", _full(2000)),
        ])
        wide = income_mod.load_icmm(cfg, LOG)
        assert set(wide["cvegeo"]) == {"01001", "01002"}, "aggregates not dropped"
        row = wide.set_index("cvegeo").loc["01001"]
        assert row["hh_income"] == 1000
        assert row["hh_income_se"] == 10
        assert row["hh_income_ci_low"] == 950
        assert row["hh_income_ci_high"] == 1050
        assert row["hh_income_cv"] == 2.0

    def test_missing_point_estimate_raises(self, income_mod, tmp_path):
        cfg = _income_config(tmp_path)
        # est 1 (Valor) absent for 01001
        _write_icmm(cfg["income"]["file"], [
            ("1", "1", {"2": 10.0, "3": 900.0, "4": 1100.0, "5": 1.0}),
        ])
        with pytest.raises(Exception, match="point estimate|est==1"):
            income_mod.load_icmm(cfg, LOG)

    def test_nonpositive_income_raises(self, income_mod, tmp_path):
        cfg = _income_config(tmp_path)
        _write_icmm(cfg["income"]["file"], [("1", "1", _full(0.0))])
        with pytest.raises(Exception, match="<= 0|positive"):
            income_mod.load_icmm(cfg, LOG)

    def test_unknown_est_code_raises(self, income_mod, tmp_path):
        cfg = _income_config(tmp_path)
        _write_icmm(cfg["income"]["file"], [
            ("1", "1", {"1": 1000.0, "2": 10.0, "3": 950.0, "4": 1050.0,
                        "5": 1.0, "9": 42.0}),
        ])
        with pytest.raises(Exception, match="est"):
            income_mod.load_icmm(cfg, LOG)


class TestHarmonizeFold:
    def test_split_child_is_population_weighted_not_summed(self, income_mod, tmp_path):
        """
        The load-bearing test. Parent P=05001 (income 5000, pop 90k) absorbs
        child C=05002 (income 3000, pop 10k). The folded mean must be the
        population-weighted average

            (5000*90000 + 3000*10000) / 100000 = 4800

        NOT the sum (8000) and NOT the unweighted mean (4000).
        """
        cfg = _income_config(tmp_path, weight="population")
        _write_crosswalk(cfg["geometry"]["crosswalk_file"], [("05002", "05001")])
        _write_iter(cfg["population"]["census_2020_tabulado"],
                    {"05001": 90000, "05002": 10000})
        wide = pd.DataFrame({
            "cvegeo": pd.Series(["05001", "05002", "01001"], dtype="string"),
            "hh_income": [5000.0, 3000.0, 1000.0],
            "hh_income_se": [50.0, 30.0, 10.0],
            "hh_income_ci_low": [4900.0, 2900.0, 950.0],
            "hh_income_ci_high": [5100.0, 3100.0, 1050.0],
            "hh_income_cv": [1.0, 1.0, 1.0],
        })
        out = income_mod.harmonize_income(wide, cfg, LOG).set_index("cvegeo")

        assert "05002" not in out.index, "child was not folded away"
        folded = out.loc["05001"]
        assert folded["hh_income"] == pytest.approx(4800.0)
        assert folded["hh_income_source"] == "icmm_2020_harmonized"
        # a weighted mean of two SAE estimates has no clean combined error
        for c in ("hh_income_se", "hh_income_ci_low", "hh_income_ci_high",
                  "hh_income_cv"):
            assert pd.isna(folded[c]), f"{c} should be NA for a folded municipality"
        # ln is of the folded value
        assert folded["log_hh_income"] == pytest.approx(np.log(4800.0))

    def test_unaffected_municipalities_keep_their_uncertainty(self, income_mod, tmp_path):
        cfg = _income_config(tmp_path, weight="population")
        _write_crosswalk(cfg["geometry"]["crosswalk_file"], [("05002", "05001")])
        _write_iter(cfg["population"]["census_2020_tabulado"],
                    {"05001": 90000, "05002": 10000})
        wide = pd.DataFrame({
            "cvegeo": pd.Series(["05001", "05002", "01001"], dtype="string"),
            "hh_income": [5000.0, 3000.0, 1000.0],
            "hh_income_se": [50.0, 30.0, 10.0],
            "hh_income_ci_low": [4900.0, 2900.0, 950.0],
            "hh_income_ci_high": [5100.0, 3100.0, 1050.0],
            "hh_income_cv": [1.0, 1.0, 2.0],
        })
        out = income_mod.harmonize_income(wide, cfg, LOG).set_index("cvegeo")
        untouched = out.loc["01001"]
        assert untouched["hh_income"] == 1000.0
        assert untouched["hh_income_source"] == "icmm_2020"
        assert untouched["hh_income_se"] == 10.0
        assert untouched["hh_income_cv"] == 2.0

    def test_no_splits_is_a_clean_passthrough(self, income_mod, tmp_path):
        cfg = _income_config(tmp_path, weight="population")
        _write_crosswalk(cfg["geometry"]["crosswalk_file"], [("99999", "99998")])
        _write_iter(cfg["population"]["census_2020_tabulado"], {"01001": 1})
        wide = pd.DataFrame({
            "cvegeo": pd.Series(["01001", "01002"], dtype="string"),
            "hh_income": [1000.0, 2000.0],
            "hh_income_se": [10.0, 20.0],
            "hh_income_ci_low": [950.0, 1900.0],
            "hh_income_ci_high": [1050.0, 2100.0],
            "hh_income_cv": [1.0, 1.0],
        })
        out = income_mod.harmonize_income(wide, cfg, LOG).set_index("cvegeo")
        assert set(out.index) == {"01001", "01002"}
        assert (out["hh_income_source"] == "icmm_2020").all()
        assert out["hh_income_cv"].notna().all()
        assert str(out.index.dtype) == "string"
        # squares: level-square and squared-log, matching the gdppc convention
        assert out.loc["01001", "hh_income_sq"] == pytest.approx(1000.0 ** 2)
        assert out.loc["01002", "hh_income_sq"] == pytest.approx(2000.0 ** 2)
        assert out.loc["01001", "log_hh_income_sq"] == pytest.approx(
            np.log(1000.0) ** 2)
