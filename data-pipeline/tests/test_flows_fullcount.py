"""
Tests for full-count (Cuestionario Basico) vs sample (Cuestionario Ampliado) mode.

CONTEXT
-------
INEGI's study description for the 2020 census confirms:
  * the Cuestionario Basico was an exhaustive enumeration of all inhabited
    dwellings, and the released microdata covers the entire enumerated population
  * FACTOR exists only in the sample tables (Viviendas_CA, Personas_CA, Migrantes)

So full-count mode must not require FACTOR, must weight every record at exactly
1.0, and must never mix the two instruments -- reading both would double-count
every sampled dwelling.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from conftest import load_step

from common import PipelineError

flows = load_step("01_flows")

LOG = logging.getLogger("test-flows")
LOG.addHandler(logging.NullHandler())


def _cfg(full: bool, **over):
    cfg = {
        "flows": {
            "use_basic_questionnaire": full,
            "min_age": 5,
            "vars": {"factor": "FACTOR", "ent_current": "ENT", "mun_current": "MUN",
                     "ent_res_5a": "ENT_PAIS_RES_5A", "mun_res_5a": "MUN_RES_5A",
                     "age": "EDAD"},
        },
    }
    cfg["flows"].update(over)
    return cfg


def _records(n=4, factor="10"):
    return pd.DataFrame({
        "orig": pd.Series(["01001"] * n, dtype="string"),
        "dest": pd.Series(["09017"] * n, dtype="string"),
        "factor": pd.Series([factor] * n, dtype="string"),
    })


class TestModeDetection:
    def test_full_count_detected(self):
        assert flows.is_full_count(_cfg(True)) is True

    def test_sample_detected(self):
        assert flows.is_full_count(_cfg(False)) is False

    def test_defaults_to_sample_when_unset(self):
        """Absent config must not silently claim a full count it does not have."""
        assert flows.is_full_count({"flows": {}}) is False


class TestRequiredVars:
    def test_factor_not_required_on_full_count(self):
        """
        The CB has no FACTOR column. Requiring it would fail the header
        assertion on a file that is perfectly correct.
        """
        got = flows._required_vars(_cfg(True))
        assert "factor" not in got
        assert got["ent_res_5a"] == "ENT_PAIS_RES_5A"
        assert got["mun_res_5a"] == "MUN_RES_5A"

    def test_factor_required_on_sample(self):
        assert flows._required_vars(_cfg(False))["factor"] == "FACTOR"

    def test_age_dropped_when_no_min_age(self):
        assert "age" not in flows._required_vars(_cfg(True, min_age=None))

    def test_migration_vars_present_in_both_modes(self):
        for full in (True, False):
            got = flows._required_vars(_cfg(full))
            assert {"ent_current", "mun_current", "ent_res_5a", "mun_res_5a"} <= set(got)


class TestRecordWeights:
    def test_full_count_weights_are_exactly_one(self):
        w = flows.record_weights(_records(), _cfg(True), LOG)
        assert (w == 1.0).all()
        assert w.dtype == "float64"

    def test_full_count_ignores_any_factor_column_present(self):
        """
        Even if a FACTOR column is somehow present, full-count mode must not use
        it -- otherwise `migrants` silently stops being a headcount.
        """
        w = flows.record_weights(_records(factor="999"), _cfg(True), LOG)
        assert (w == 1.0).all()

    def test_sample_uses_the_factor_column(self):
        w = flows.record_weights(_records(factor="10"), _cfg(False), LOG)
        assert (w == 10.0).all()

    def test_sample_rejects_non_numeric_factor(self):
        bad = _records()
        bad.loc[0, "factor"] = "not-a-number"
        with pytest.raises(PipelineError, match="non-numeric expansion FACTOR"):
            flows.record_weights(bad, _cfg(False), LOG)

    def test_error_suggests_full_count_mode(self):
        """A missing/garbage FACTOR usually means these are CB files."""
        bad = _records()
        bad.loc[0, "factor"] = ""
        with pytest.raises(PipelineError) as exc:
            flows.record_weights(bad, _cfg(False), LOG)
        assert "use_basic_questionnaire" in str(exc.value)


class TestAggregation:
    def test_full_count_migrants_equals_record_count(self):
        out = flows.aggregate_od(_records(n=7), _cfg(True), LOG, "test")
        assert len(out) == 1
        assert out.loc[0, "migrants"] == 7.0
        assert out.loc[0, "migrants_unweighted"] == 7

    def test_sample_scales_by_factor(self):
        out = flows.aggregate_od(_records(n=3, factor="10"), _cfg(False), LOG, "test")
        assert out.loc[0, "migrants"] == 30.0
        assert out.loc[0, "migrants_unweighted"] == 3

    def test_full_count_invariant_is_enforced(self, monkeypatch):
        """
        The migrants == migrants_unweighted invariant is a real guard, not a
        comment. If a weight leaks in, the build must stop.
        """
        monkeypatch.setattr(
            flows, "record_weights",
            lambda df, cfg, log: pd.Series(2.0, index=df.index, dtype="float64"))
        with pytest.raises(PipelineError, match="weighted and unweighted counts disagree"):
            flows.aggregate_od(_records(n=3), _cfg(True), LOG, "test")

    def test_separate_dyads_stay_separate(self):
        recs = pd.DataFrame({
            "orig": pd.Series(["01001", "01001", "09017"], dtype="string"),
            "dest": pd.Series(["09017", "09017", "01001"], dtype="string"),
            "factor": pd.Series(["1", "1", "1"], dtype="string"),
        })
        out = flows.aggregate_od(recs, _cfg(True), LOG, "test").sort_values("orig")
        assert len(out) == 2
        assert out[out["orig"] == "01001"]["migrants"].iat[0] == 2.0
        assert out[out["orig"] == "09017"]["migrants"].iat[0] == 1.0

    def test_keys_remain_strings_with_leading_zeros(self):
        out = flows.aggregate_od(_records(), _cfg(True), LOG, "test")
        assert str(out["orig"].dtype) == "string"
        assert out.loc[0, "orig"] == "01001"


class TestInstrumentSelection:
    """
    Reading CB and CA files together would double-count every sampled dwelling.
    The split keys on the "_CA" marker in the filename, and whatever is
    excluded is reported rather than dropped quietly.
    """

    def _setup(self, tmp_path, names):
        d = tmp_path / "censo2020"
        d.mkdir(parents=True, exist_ok=True)
        for n in names:
            (d / n).write_text("ENT,MUN\n01,001\n", encoding="utf-8")
        return d

    def _cfg_glob(self, full):
        return _cfg(full, microdata_is_sharded=True,
                    microdata_glob="censo2020/Personas*.csv")

    def test_full_count_keeps_only_cb_when_both_present(self, tmp_path, monkeypatch):
        self._setup(tmp_path, ["Personas.csv", "Personas01.csv", "Personas_CA.csv"])
        monkeypatch.setattr(flows, "ROOT", tmp_path)
        got = [f.name for f in flows._microdata_files(self._cfg_glob(True), LOG)]
        assert sorted(got) == ["Personas.csv", "Personas01.csv"]
        assert "Personas_CA.csv" not in got

    def test_sample_keeps_only_ca_when_both_present(self, tmp_path, monkeypatch):
        self._setup(tmp_path, ["Personas.csv", "Personas_CA.csv", "Personas_CA01.csv"])
        monkeypatch.setattr(flows, "ROOT", tmp_path)
        got = [f.name for f in flows._microdata_files(self._cfg_glob(False), LOG)]
        assert sorted(got) == ["Personas_CA.csv", "Personas_CA01.csv"]

    def test_plain_personas_csv_is_matched(self, tmp_path, monkeypatch):
        """
        Regression: an earlier glob (`Personas[!_]*.csv`) failed to match the
        plain national file `Personas.csv`, which is the main CB download.
        """
        self._setup(tmp_path, ["Personas.csv"])
        monkeypatch.setattr(flows, "ROOT", tmp_path)
        got = flows._microdata_files(self._cfg_glob(True), LOG)
        assert [f.name for f in got] == ["Personas.csv"]

    def test_full_count_with_only_ca_files_raises_and_suggests_the_fix(
            self, tmp_path, monkeypatch):
        self._setup(tmp_path, ["Personas_CA.csv"])
        monkeypatch.setattr(flows, "ROOT", tmp_path)
        with pytest.raises(PipelineError) as exc:
            flows._microdata_files(self._cfg_glob(True), LOG)
        assert "use_basic_questionnaire: false" in str(exc.value)

    def test_sample_with_only_cb_files_raises_and_suggests_the_fix(
            self, tmp_path, monkeypatch):
        self._setup(tmp_path, ["Personas.csv"])
        monkeypatch.setattr(flows, "ROOT", tmp_path)
        with pytest.raises(PipelineError) as exc:
            flows._microdata_files(self._cfg_glob(False), LOG)
        assert "use_basic_questionnaire: true" in str(exc.value)

    def test_no_match_raises_with_a_useful_message(self, tmp_path, monkeypatch):
        self._setup(tmp_path, [])
        monkeypatch.setattr(flows, "ROOT", tmp_path)
        cfg = _cfg(True, microdata_is_sharded=True,
                   microdata_glob="censo2020/Nothing*.csv")
        with pytest.raises(PipelineError, match="Cuestionario BASICO"):
            flows._microdata_files(cfg, LOG)


class TestShippedConfig:
    def test_config_is_set_to_sample_mode(self):
        """
        The shipped config uses the CA sample -- guard against a silent flip.

        This asserted full-count mode until 2026-07-21, when it was established
        that INEGI does NOT publicly release the full-count CB microdata: the
        files on the microdatos page are explicitly examples ("no permiten
        hacer ningun tipo de inferencia"), and the real full count is available
        only through the Laboratorio de microdatos. The public instrument is
        the CA ~10% sample (Censo2020_CA_eum_csv.zip), FACTOR-weighted. If you
        obtain the CB through the lab, flip the config and this test together.
        """
        from common import load_config
        assert load_config()["flows"]["use_basic_questionnaire"] is False

    def test_glob_matches_both_instruments_for_code_to_split(self):
        """
        The glob is intentionally broad; the instrument split happens in code.
        If someone narrows it here, the split logic silently stops being
        exercised, so pin the intent.
        """
        import fnmatch
        from common import load_config
        leaf = load_config()["flows"]["microdata_glob"].rsplit("/", 1)[-1]
        for name in ("Personas.csv", "Personas01.csv", "Personas_CA.csv"):
            assert fnmatch.fnmatch(name, leaf), f"{leaf!r} should match {name}"
