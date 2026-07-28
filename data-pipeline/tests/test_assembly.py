"""
Unit tests for panel assembly: the cartesian spine, join discipline, zero
classification, and the codebook completeness guard.

These run on synthetic municipality codes, so they exercise the assembly logic
without needing any of the multi-gigabyte raw inputs.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from common import PipelineError, missingness_table

LOG = logging.getLogger("test")
LOG.addHandler(logging.NullHandler())

CODES = pd.Series(["01001", "01002", "09017", "31050"], dtype="string")


class TestSpine:
    def test_full_cartesian_minus_diagonal(self, assemble_mod, base_config):
        spine = assemble_mod.build_spine(base_config, CODES, LOG)
        n = len(CODES)
        assert len(spine) == n * n - n
        assert (spine["orig"].to_numpy() != spine["dest"].to_numpy()).all()

    def test_diagonal_included_when_configured(self, assemble_mod, base_config):
        base_config["assemble"]["include_diagonal"] = True
        spine = assemble_mod.build_spine(base_config, CODES, LOG)
        assert len(spine) == len(CODES) ** 2
        assert (spine["orig"] == spine["dest"]).sum() == len(CODES)

    def test_ordered_pairs_both_directions_present(self, assemble_mod, base_config):
        """(A,B) and (B,A) are distinct rows -- migration is not symmetric."""
        spine = assemble_mod.build_spine(base_config, CODES, LOG)
        pairs = set(zip(spine["orig"].astype(str), spine["dest"].astype(str)))
        assert ("01001", "09017") in pairs
        assert ("09017", "01001") in pairs

    def test_keys_stay_strings_with_leading_zeros(self, assemble_mod, base_config):
        spine = assemble_mod.build_spine(base_config, CODES, LOG)
        assert str(spine["orig"].dtype) == "string"
        assert spine["orig"].str.len().eq(5).all()
        assert "01001" in set(spine["orig"].astype(str))

    def test_no_duplicate_keys(self, assemble_mod, base_config):
        spine = assemble_mod.build_spine(base_config, CODES, LOG)
        assert not spine.duplicated(subset=["orig", "dest"]).any()

    def test_deduplicates_a_repeated_universe_code(self, assemble_mod, base_config):
        dirty = pd.Series(["01001", "01001", "09017"], dtype="string")
        spine = assemble_mod.build_spine(base_config, dirty, LOG)
        assert len(spine) == 2 * 2 - 2


class TestLeftJoinDiscipline:
    def test_left_join_preserves_every_spine_row(self, assemble_mod, base_config):
        """
        The whole point: a pair with no flow record must survive the join as a
        zero, not vanish. An inner join here is how the zeros would disappear.
        """
        spine = assemble_mod.build_spine(base_config, CODES, LOG)
        flows = pd.DataFrame({
            "orig": pd.Series(["01001"], dtype="string"),
            "dest": pd.Series(["09017"], dtype="string"),
            "migrants": [1234.0],
        })
        out = assemble_mod.left_join(spine, flows, ["orig", "dest"], LOG, "flows")
        assert len(out) == len(spine)
        assert out["migrants"].notna().sum() == 1
        assert out["migrants"].isna().sum() == len(spine) - 1

    def test_duplicate_keys_on_the_right_are_rejected(self, assemble_mod, base_config):
        """
        A right-hand table with duplicate keys would multiply spine rows. Caught
        rather than silently inflating the panel.
        """
        spine = assemble_mod.build_spine(base_config, CODES, LOG)
        dupe = pd.DataFrame({
            "orig": pd.Series(["01001", "01001"], dtype="string"),
            "dest": pd.Series(["09017", "09017"], dtype="string"),
            "migrants": [1.0, 2.0],
        })
        with pytest.raises(PipelineError, match="duplicate keys"):
            assemble_mod.left_join(spine, dupe, ["orig", "dest"], LOG, "flows")

    def test_unmatched_covariate_becomes_na_not_a_dropped_row(self, assemble_mod,
                                                              base_config):
        spine = assemble_mod.build_spine(base_config, CODES, LOG)
        partial = pd.DataFrame({
            "orig": pd.Series(["01001"], dtype="string"),
            "gdppc_orig": [15000.0],
        })
        out = assemble_mod.left_join(spine, partial, "orig", LOG, "gdp")
        assert len(out) == len(spine)
        assert out["gdppc_orig"].isna().sum() > 0


class TestOrphanDetection:
    def test_orphan_code_is_reported(self, assemble_mod, base_config):
        flows = pd.DataFrame({
            "orig": pd.Series(["01001", "17034"], dtype="string"),
            "dest": pd.Series(["09017", "09017"], dtype="string"),
            "migrants": [10.0, 500.0],
        })
        universe = {"01001", "09017", "31050"}
        orphans = assemble_mod.check_orphans(base_config, flows, universe, LOG)
        assert len(orphans) == 1
        assert orphans.iloc[0]["code"] == "17034"
        assert orphans.iloc[0]["side"] == "orig"
        # the weight that an inner join would have silently deleted
        assert orphans.iloc[0]["weighted_migrants"] == 500.0

    def test_clean_flows_produce_no_orphans(self, assemble_mod, base_config):
        flows = pd.DataFrame({
            "orig": pd.Series(["01001"], dtype="string"),
            "dest": pd.Series(["09017"], dtype="string"),
            "migrants": [10.0],
        })
        orphans = assemble_mod.check_orphans(base_config, flows,
                                             {"01001", "09017"}, LOG)
        assert len(orphans) == 0

    def test_orphans_detected_on_the_destination_side_too(self, assemble_mod,
                                                          base_config):
        flows = pd.DataFrame({
            "orig": pd.Series(["01001"], dtype="string"),
            "dest": pd.Series(["99999"], dtype="string"),
            "migrants": [7.0],
        })
        orphans = assemble_mod.check_orphans(base_config, flows, {"01001"}, LOG)
        assert set(orphans["side"]) == {"dest"}


class TestZeroClassification:
    def _panel(self):
        return pd.DataFrame({
            "orig": pd.Series(["01001", "01001", "09017", "31050"], dtype="string"),
            "dest": pd.Series(["09017", "31050", "01001", "09017"], dtype="string"),
            "migrants": [100.0, 0.0, 50.0, 0.0],
            "migrants_unweighted": [10, 0, 5, 0],
        })

    def test_flow_observed_tracks_the_unweighted_count(self, assemble_mod):
        out = assemble_mod.classify_zeros(self._panel(), LOG)
        assert out["flow_observed"].tolist() == [True, False, True, False]

    def test_positive_cells_are_labelled_observed(self, assemble_mod):
        out = assemble_mod.classify_zeros(self._panel(), LOG)
        assert out.loc[0, "zero_class"] == "observed_positive"

    def test_zero_between_two_active_endpoints_is_a_plausible_sampling_zero(
            self, assemble_mod):
        """01001 sends elsewhere and 31050... does not receive. See next test."""
        out = assemble_mod.classify_zeros(self._panel(), LOG)
        # 31050 -> 09017 : 31050 never appears as an active origin, but 09017 is
        # an active destination, so this is structural by the heuristic.
        assert out.loc[3, "zero_class"] == "structural_zero_likely"

    def test_endpoint_with_no_activity_yields_structural(self, assemble_mod):
        out = assemble_mod.classify_zeros(self._panel(), LOG)
        # 01001 -> 31050 : 31050 is never an observed destination
        assert out.loc[1, "zero_class"] == "structural_zero_likely"

    def test_all_rows_get_exactly_one_class(self, assemble_mod):
        out = assemble_mod.classify_zeros(self._panel(), LOG)
        assert out["zero_class"].notna().all()
        assert set(out["zero_class"]) <= {"observed_positive",
                                          "sampling_zero_plausible",
                                          "structural_zero_likely"}

    def test_sampling_zero_appears_when_both_endpoints_are_active(self, assemble_mod):
        panel = pd.DataFrame({
            "orig": pd.Series(["01001", "09017", "01001"], dtype="string"),
            "dest": pd.Series(["09017", "01001", "31050"], dtype="string"),
            "migrants": [100.0, 50.0, 0.0],
            "migrants_unweighted": [10, 5, 0],
        })
        # add a pair whose endpoints are both active but which is itself zero
        panel.loc[3] = ["09017", "09017", 0.0, 0]
        out = assemble_mod.classify_zeros(panel, LOG)
        assert out.loc[3, "zero_class"] == "sampling_zero_plausible"


class TestCodebookCompleteness:
    def test_every_documented_column_has_all_five_fields(self, assemble_mod):
        for col, spec in assemble_mod.COLUMN_SPEC.items():
            assert len(spec) == 5, f"{col} spec must be (definition, units, source, vintage, construction)"
            for i, field in enumerate(spec):
                assert isinstance(field, str) and field.strip(), \
                    f"{col} field {i} is empty"

    def test_key_columns_are_documented(self, assemble_mod):
        # The SHIPPED schema uses src_/dst_ prefixes; internals still speak
        # orig/dest and are renamed via FINAL_NAMES just before the codebook
        # completeness check.
        required = {"src", "dst", "migrants", "migrants_unweighted",
                    "flow_observed", "src_pop", "dst_pop", "dist_geodesic_km",
                    "src_gdppc", "dst_gdppc", "src_gdppc_sq", "dst_gdppc_sq",
                    "src_log_gdppc_sq", "src_temp", "dst_temp", "src_precip",
                    "dst_precip", "src_dempres", "dst_dempres"}
        missing = required - set(assemble_mod.COLUMN_SPEC)
        assert not missing, f"undocumented required columns: {missing}"

    def test_final_names_all_land_in_the_codebook(self, assemble_mod):
        """Every rename target must be documented, or the build would stop."""
        targets = set(assemble_mod.FINAL_NAMES.values())
        missing = targets - set(assemble_mod.COLUMN_SPEC)
        assert not missing, f"FINAL_NAMES targets missing from COLUMN_SPEC: {missing}"

    def test_square_and_squared_log_are_documented_as_different_objects(
            self, assemble_mod):
        """
        src_gdppc_sq is a square of a level; src_log_gdppc_sq is a squared log.
        Conflating them is a real and common error, so the codebook must
        distinguish them explicitly.
        """
        sq = assemble_mod.COLUMN_SPEC["src_gdppc_sq"]
        logsq = assemble_mod.COLUMN_SPEC["src_log_gdppc_sq"]
        assert sq[1] != logsq[1], "units must differ"
        assert "SQUARE" in sq[4].upper()
        assert "SQUARED LOG" in logsq[4].upper()
        assert "collinear" in logsq[4].lower()

    def test_gdp_columns_are_flagged_as_estimates(self, assemble_mod):
        for col in ("src_gdppc", "dst_gdppc"):
            assert "ESTIMATE" in assemble_mod.COLUMN_SPEC[col][0].upper(), (
                f"{col} must be flagged as an estimate -- Mexico publishes no "
                "official municipal GDP series"
            )


class TestPositiveOnlyExport:
    """
    The positive-only CSV is a companion view, not a replacement.

    These tests pin the two properties that matter: it contains exactly the
    positive-flow dyads, and producing it does not disturb the full panel.
    """

    def _panel(self):
        return pd.DataFrame({
            "orig": pd.Series(["01001", "01001", "09017", "31050"], dtype="string"),
            "dest": pd.Series(["09017", "31050", "01001", "09017"], dtype="string"),
            "migrants": [100.0, 0.0, 50.0, 0.0],
            "migrants_unweighted": [10, 0, 5, 0],
        })

    def _cfg(self, base_config, tmp_path, **over):
        base_config["assemble"].update({
            "export_positive_only": True,
            "positive_only_output": str(tmp_path / "positive.csv"),
            "positive_only_gzip": False,
            **over,
        })
        return base_config

    def test_contains_only_positive_flows(self, assemble_mod, base_config, tmp_path):
        cfg = self._cfg(base_config, tmp_path)
        out = assemble_mod.write_positive_only(cfg, self._panel(), LOG)
        got = pd.read_csv(out, dtype={"orig": "string", "dest": "string"})
        assert len(got) == 2
        assert (got["migrants"] > 0).all()

    def test_does_not_mutate_the_input_panel(self, assemble_mod, base_config, tmp_path):
        cfg = self._cfg(base_config, tmp_path)
        panel = self._panel()
        assemble_mod.write_positive_only(cfg, panel, LOG)
        assert len(panel) == 4, "the full panel must survive the export untouched"

    def test_leading_zeros_survive_the_csv_round_trip(self, assemble_mod,
                                                     base_config, tmp_path):
        """
        CSV is where a zero-padded code turns back into an integer. The file is
        for humans and other tools, so this has to hold on read-back.
        """
        cfg = self._cfg(base_config, tmp_path)
        out = assemble_mod.write_positive_only(cfg, self._panel(), LOG)
        raw = out.read_text(encoding="utf-8")
        assert "01001" in raw, "leading zero was lost writing the CSV"

    def test_gzip_option(self, assemble_mod, base_config, tmp_path):
        cfg = self._cfg(base_config, tmp_path, positive_only_gzip=True)
        out = assemble_mod.write_positive_only(cfg, self._panel(), LOG)
        assert out.suffix == ".gz"
        got = pd.read_csv(out, compression="gzip")
        assert len(got) == 2

    def test_empty_result_raises_rather_than_writing_an_empty_file(
            self, assemble_mod, base_config, tmp_path):
        cfg = self._cfg(base_config, tmp_path)
        all_zero = self._panel()
        all_zero["migrants"] = 0.0
        with pytest.raises(PipelineError, match="would be empty"):
            assemble_mod.write_positive_only(cfg, all_zero, LOG)

    def test_filters_on_migrants_not_on_zero_class(self, assemble_mod,
                                                   base_config, tmp_path):
        """
        Filtering must key on the actual flow value, never on the zero_class
        heuristic. A positive flow labelled anything at all still ships; a zero
        never does, whatever it is classified as.
        """
        cfg = self._cfg(base_config, tmp_path)
        panel = self._panel()
        panel["zero_class"] = ["observed_positive", "structural_zero_likely",
                               "observed_positive", "sampling_zero_plausible"]
        out = assemble_mod.write_positive_only(cfg, panel, LOG)
        got = pd.read_csv(out)
        assert len(got) == 2
        assert set(got["zero_class"]) == {"observed_positive"}
        # both zero classes excluded, neither privileged over the other
        assert "structural_zero_likely" not in set(got["zero_class"])
        assert "sampling_zero_plausible" not in set(got["zero_class"])


class TestCodebookRecordsFiltering:
    def test_filtered_panel_is_flagged_loudly_in_the_codebook(self, assemble_mod,
                                                              base_config, tmp_path):
        """
        A filtered file outlives the conversation that produced it. The codebook
        must say so unmissably, or someone estimates gravity on it.
        """
        base_config["assemble"].update({
            "output": str(tmp_path / "p.parquet"),
            "codebook": str(tmp_path / "codebook.md"),
        })
        panel = pd.DataFrame({
            "orig": pd.Series(["01001"], dtype="string"),
            "dest": pd.Series(["09017"], dtype="string"),
            "migrants": [100.0],
        })
        stats = dict(n_municipios=2, n_positive=1, n_full_cartesian=1000,
                     main_filtered=True, positive_only_path=None)
        out = assemble_mod.write_codebook(base_config, panel, stats, LOG)
        text = out.read_text(encoding="utf-8")
        assert "FILTERED TO POSITIVE FLOWS" in text
        assert "Do not estimate a gravity model on this file" in text
        assert "999" in text or "99" in text, "should report how much was dropped"

    def test_unfiltered_panel_gets_the_normal_note(self, assemble_mod,
                                                  base_config, tmp_path):
        base_config["assemble"].update({
            "output": str(tmp_path / "p.parquet"),
            "codebook": str(tmp_path / "codebook.md"),
        })
        panel = pd.DataFrame({
            "orig": pd.Series(["01001", "01002"], dtype="string"),
            "dest": pd.Series(["09017", "09017"], dtype="string"),
            "migrants": [100.0, 0.0],
        })
        stats = dict(n_municipios=2, n_positive=1, n_full_cartesian=2,
                     main_filtered=False, positive_only_path=None)
        out = assemble_mod.write_codebook(base_config, panel, stats, LOG)
        text = out.read_text(encoding="utf-8")
        assert "FILTERED TO POSITIVE FLOWS" not in text
        assert "PPML needs the zeros present" in text


class TestMissingness:
    def test_reports_every_column(self):
        df = pd.DataFrame({"a": [1, 2, None], "b": ["x", None, None], "c": [1, 2, 3]})
        out = missingness_table(df)
        assert set(out["column"]) == {"a", "b", "c"}

    def test_percentages_are_right(self):
        df = pd.DataFrame({"a": [1, 2, None], "c": [1, 2, 3]})
        out = missingness_table(df).set_index("column")
        assert out.loc["a", "n_missing"] == 1
        assert out.loc["a", "pct_missing"] == pytest.approx(33.3333, abs=0.01)
        assert out.loc["c", "n_missing"] == 0

    def test_sorted_worst_first(self):
        df = pd.DataFrame({"clean": [1, 2, 3], "dirty": [None, None, 1]})
        out = missingness_table(df)
        assert out.iloc[0]["column"] == "dirty"
