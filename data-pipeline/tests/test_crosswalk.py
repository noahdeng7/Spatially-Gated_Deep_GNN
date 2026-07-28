"""
Unit tests for the 2015->2020 municipality crosswalk and CVEGEO handling.

The crosswalk tests are built around known municipality splits. Note that the
seed crosswalk ships with `verified=False` on every row -- the CODES in it have
not been checked against the INEGI catalogo. These tests therefore assert the
crosswalk MECHANISM (a child maps to its parent, aggregation is balanced,
chained splits are rejected) using synthetic fixtures, and separately assert
that the seed data is well-FORMED without asserting that its codes are correct.

Testing the mechanism against synthetic data is deliberate: if the tests hard-
coded the real codes, verifying and correcting a code would break the test
suite, which trains people to edit tests rather than fix data.
"""

from __future__ import annotations

import pandas as pd
import pytest

import geo
from common import PipelineError, cvegeo, cvegeo_series


# --- synthetic splits, standing in for the real ones ------------------------
# Shaped exactly like the Morelos 2017 and Quintana Roo 2016 cases: one parent
# gaining one child, and one parent gaining two.
SYNTHETIC_SPLITS = pd.DataFrame([
    dict(cvegeo_child="17034", name_child="Child A", cvegeo_parent="17020",
         name_parent="Parent A", year_created=2017, verified=False, source_note="test"),
    dict(cvegeo_child="17035", name_child="Child B", cvegeo_parent="17026",
         name_parent="Parent B", year_created=2017, verified=False, source_note="test"),
    dict(cvegeo_child="23011", name_child="Child C", cvegeo_parent="23005",
         name_parent="Parent C", year_created=2016, verified=False, source_note="test"),
]).astype({"cvegeo_child": "string", "cvegeo_parent": "string"})


class TestApplyCrosswalk:
    def test_child_maps_to_parent(self):
        codes = pd.Series(["17034", "17035", "23011"], dtype="string")
        out = geo.apply_crosswalk(codes, SYNTHETIC_SPLITS, "aggregate_to_parent")
        assert out.tolist() == ["17020", "17026", "23005"]

    def test_unaffected_codes_pass_through_unchanged(self):
        """The ~2,460 municipalities that never changed must be untouched."""
        codes = pd.Series(["01001", "09017", "15106", "31050"], dtype="string")
        out = geo.apply_crosswalk(codes, SYNTHETIC_SPLITS, "aggregate_to_parent")
        assert out.tolist() == codes.tolist()

    def test_parent_codes_are_idempotent(self):
        """Applying the crosswalk to an already-harmonized series is a no-op."""
        codes = pd.Series(["17020", "17026", "23005"], dtype="string")
        once = geo.apply_crosswalk(codes, SYNTHETIC_SPLITS, "aggregate_to_parent")
        twice = geo.apply_crosswalk(once, SYNTHETIC_SPLITS, "aggregate_to_parent")
        assert once.tolist() == twice.tolist() == codes.tolist()

    def test_output_is_still_string_dtype(self):
        codes = pd.Series(["17034", "01001"], dtype="string")
        out = geo.apply_crosswalk(codes, SYNTHETIC_SPLITS, "aggregate_to_parent")
        assert str(out.dtype) == "string"
        assert out.iloc[1] == "01001", "leading zero survived the crosswalk"

    def test_nulls_survive(self):
        codes = pd.Series(["17034", pd.NA, "01001"], dtype="string")
        out = geo.apply_crosswalk(codes, SYNTHETIC_SPLITS, "aggregate_to_parent")
        assert out.iloc[0] == "17020"
        assert pd.isna(out.iloc[1])

    def test_strategy_none_is_a_passthrough(self):
        codes = pd.Series(["17034"], dtype="string")
        out = geo.apply_crosswalk(codes, SYNTHETIC_SPLITS, "none")
        assert out.tolist() == ["17034"]

    def test_allocate_strategy_refuses_rather_than_guessing(self):
        """
        'allocate' is documented but not implemented, and it raises rather than
        quietly falling back to aggregation -- a silent fallback would mean the
        config said one thing and the data another.
        """
        codes = pd.Series(["17034"], dtype="string")
        with pytest.raises(NotImplementedError, match="allocate"):
            geo.apply_crosswalk(codes, SYNTHETIC_SPLITS, "allocate")

    def test_unknown_strategy_raises(self):
        codes = pd.Series(["17034"], dtype="string")
        with pytest.raises(PipelineError, match="unknown crosswalk_strategy"):
            geo.apply_crosswalk(codes, SYNTHETIC_SPLITS, "sideways")


class TestPanelBalance:
    def test_aggregation_conserves_population(self):
        """
        Population is extensive: folding children into a parent must conserve
        the total. If it does not, the panel's denominators are wrong.
        """
        pop = pd.DataFrame({
            "cvegeo": pd.Series(["17020", "17034", "17026", "17035", "01001"],
                                dtype="string"),
            "pop": [50_000, 12_000, 8_000, 3_000, 900_000],
        })
        total_before = pop["pop"].sum()
        pop["cvegeo"] = geo.apply_crosswalk(pop["cvegeo"], SYNTHETIC_SPLITS,
                                            "aggregate_to_parent")
        agg = pop.groupby("cvegeo", as_index=False)["pop"].sum()
        assert agg["pop"].sum() == total_before
        assert agg.loc[agg["cvegeo"] == "17020", "pop"].iat[0] == 62_000
        assert agg.loc[agg["cvegeo"] == "17026", "pop"].iat[0] == 11_000

    def test_both_sides_harmonized_keeps_the_panel_square(self):
        """
        Harmonizing only one side of the dyad would leave the origin and
        destination universes different sizes and the panel non-square.
        """
        flows = pd.DataFrame({
            "orig": pd.Series(["17034", "17020", "01001"], dtype="string"),
            "dest": pd.Series(["23011", "23005", "17035"], dtype="string"),
        })
        flows["orig"] = geo.apply_crosswalk(flows["orig"], SYNTHETIC_SPLITS,
                                            "aggregate_to_parent")
        flows["dest"] = geo.apply_crosswalk(flows["dest"], SYNTHETIC_SPLITS,
                                            "aggregate_to_parent")
        assert set(flows["orig"]) == {"17020", "01001"}
        assert set(flows["dest"]) == {"23005", "17026"}
        assert "17034" not in set(flows["orig"]) | set(flows["dest"])
        assert "23011" not in set(flows["orig"]) | set(flows["dest"])

    def test_flows_between_a_child_and_its_own_parent_land_on_the_diagonal(self):
        """
        A move from a split child to its own parent was an internal move within
        the pre-split municipality. After aggregation it must land on the
        diagonal, where the pipeline routes it out of the dyadic panel -- not be
        counted as a real between-municipality flow.
        """
        flows = pd.DataFrame({
            "orig": pd.Series(["17034"], dtype="string"),
            "dest": pd.Series(["17020"], dtype="string"),
            "migrants": [500.0],
        })
        flows["orig"] = geo.apply_crosswalk(flows["orig"], SYNTHETIC_SPLITS,
                                            "aggregate_to_parent")
        flows["dest"] = geo.apply_crosswalk(flows["dest"], SYNTHETIC_SPLITS,
                                            "aggregate_to_parent")
        assert (flows["orig"] == flows["dest"]).all()


class TestCrosswalkIntegrity:
    def test_chained_splits_are_rejected(self, tmp_path, base_config):
        """
        A code that is both a child and a parent makes the mapping order-
        dependent: whether B->A happens before C->B changes the answer. Loading
        must reject it rather than pick an order.
        """
        path = tmp_path / "raw" / "crosswalk.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            dict(cvegeo_child="17035", name_child="B", cvegeo_parent="17034",
                 name_parent="A", year_created=2017, verified=True, source_note="x"),
            dict(cvegeo_child="17034", name_child="A", cvegeo_parent="17020",
                 name_parent="P", year_created=2017, verified=True, source_note="x"),
        ]).to_csv(path, index=False)

        with pytest.raises(PipelineError, match="chained splits"):
            geo.load_crosswalk(base_config)

    def test_seed_crosswalk_is_wellformed(self):
        """
        Asserts the SHAPE of the shipped seed data, not the correctness of its
        codes -- every seed row is marked verified=False precisely because the
        codes have not been checked against the INEGI catalogo.
        """
        df = pd.DataFrame(geo.SEED_CROSSWALK)
        assert len(df) > 0
        assert set(geo.CROSSWALK_COLUMNS) <= set(df.columns)
        for col in ("cvegeo_child", "cvegeo_parent"):
            assert df[col].str.len().eq(5).all(), f"{col} must be 5 characters"
            assert df[col].str.isdigit().all(), f"{col} must be all digits"
        assert df["cvegeo_child"].is_unique, "a child cannot have two parents"
        assert not set(df["cvegeo_child"]) & set(df["cvegeo_parent"]), \
            "seed crosswalk contains a chained split"

    def test_verified_rows_cite_their_evidence(self):
        """
        A guard on honesty. `verified=True` is a claim that someone checked the
        codes against a named source, so the row must say which source and when.

        These rows were verified against INEGI ITER 2020 on 2026-07-21, after an
        earlier version paired Xoxocotla with 17020 -- which is Tepoztlan, a
        different municipality entirely. That is exactly the silent corruption
        the verification exists to prevent, so the evidence gets cited.
        """
        df = pd.DataFrame(geo.SEED_CROSSWALK)
        for _, r in df[df["verified"]].iterrows():
            note = str(r["source_note"])
            assert "ITER" in note or "catalogo" in note.lower(), (
                f"row {r['cvegeo_child']} claims verified=True but its "
                "source_note does not name the source it was checked against"
            )
            assert any(ch.isdigit() for ch in note), (
                f"row {r['cvegeo_child']} claims verified=True but cites no "
                "date or code evidence"
            )

    def test_every_seed_row_carries_a_source_note(self):
        df = pd.DataFrame(geo.SEED_CROSSWALK)
        assert df["source_note"].str.len().gt(10).all()

    def test_known_correct_codes(self):
        """
        Pin the ITER-verified mappings. These are the specific values that were
        wrong before, so a regression here is worth catching loudly.
        """
        by_child = {r["cvegeo_child"]: r for r in geo.SEED_CROSSWALK}
        assert by_child["17034"]["cvegeo_parent"] == "17015"   # Coatetelco <- Miacatlan
        assert by_child["17035"]["cvegeo_parent"] == "17017"   # Xoxocotla  <- Puente de Ixtla
        assert by_child["17036"]["cvegeo_parent"] == "17022"   # Hueyapan   <- Tetela del Volcan
        assert by_child["23011"]["cvegeo_parent"] == "23005"   # Puerto Morelos <- Benito Juarez
        assert by_child["02006"]["cvegeo_parent"] == "02001"   # San Quintin <- Ensenada

    def test_wrong_parents_from_the_earlier_version_are_gone(self):
        """17020 is Tepoztlan and 17026 is Tlayacapan -- neither is a parent."""
        parents = {r["cvegeo_parent"] for r in geo.SEED_CROSSWALK}
        assert "17020" not in parents, "17020 is Tepoztlan, not Puente de Ixtla"
        assert "17026" not in parents, "17026 is Tlayacapan, not Tetela del Volcan"

    def test_san_felipe_is_a_post_census_row_not_an_in_window_one(self):
        """
        ITER 2020 lists Baja California with exactly six municipalities and no
        San Felipe, so it cannot be a 2015-2020 split -- no 2015-residence code
        can refer to it. It belongs in the crosswalk only as a POST-CENSUS row,
        to fold it out of a newer geometry edition.

        An earlier version had it as an in-window split from Mexicali, which was
        wrong about what the row is for.
        """
        row = next(r for r in geo.SEED_CROSSWALK if r["cvegeo_child"] == "02007")
        assert row["year_created"] > 2020
        assert "post-2020" in row["source_note"].lower()

    def test_crosswalk_serves_its_two_documented_purposes(self):
        """
        The crosswalk carries rows of two kinds, and both are legitimate:

          year_created <= 2020  harmonize FLOW codes -- municipalities created
                                during the migration window, whose 2015-residence
                                records must fold into the pre-split parent

          year_created >  2020  harmonize GEOMETRY -- municipalities created
                                after the census, present only because the
                                available Marco Geoestadistico edition is newer
                                than 2020. Folding them dissolves child into
                                parent and reconstructs the census boundary.

        A row outside both categories is a mistake.
        """
        df = pd.DataFrame(geo.SEED_CROSSWALK)
        in_window = df["year_created"].between(2015, 2020)
        post_census = df["year_created"] > 2020
        assert (in_window | post_census).all(), (
            "seed crosswalk contains a split before 2015; it cannot affect a "
            "2015-2020 panel and the row is unnecessary"
        )
        assert in_window.any(), "expected at least one in-window split"
        assert post_census.any(), (
            "expected post-2020 rows -- they are what allow a newer Marco "
            "Geoestadistico edition to be used"
        )

    def test_post_census_rows_cite_how_the_parent_was_derived(self):
        """
        Post-2020 parents were derived by point-in-polygon of ITER 2020
        localities, not recalled. The note must say so, because the earlier
        recalled mappings were wrong.
        """
        df = pd.DataFrame(geo.SEED_CROSSWALK)
        for _, r in df[df["year_created"] > 2020].iterrows():
            note = str(r["source_note"]).lower()
            assert "point-in-polygon" in note, (
                f"post-census row {r['cvegeo_child']} does not record how its "
                "parent was determined"
            )
            assert "localities" in note or "locality" in note

    def test_post_census_parents_are_the_derived_ones(self):
        """Pin the spatially-derived mappings against silent regression."""
        by_child = {r["cvegeo_child"]: r["cvegeo_parent"]
                    for r in geo.SEED_CROSSWALK}
        assert by_child["02007"] == "02002"   # San Felipe   <- Mexicali
        assert by_child["04013"] == "04001"   # Dzitbalche   <- Calkini
        assert by_child["12082"] == "12053"   # Las Vigas    <- San Marcos
        assert by_child["12083"] == "12012"   # Nuu Savi     <- Ayutla de los Libres
        assert by_child["12084"] == "12041"   # Sta Cruz     <- Malinaltepec
        assert by_child["12085"] == "12023"   # San Nicolas  <- Cuajinicuilapa

    def test_seeding_writes_a_file_that_loads_back(self, base_config):
        path = geo.seed_crosswalk_if_absent(base_config)
        assert path.exists()
        loaded = geo.load_crosswalk(base_config)
        assert len(loaded) == len(geo.SEED_CROSSWALK)
        assert str(loaded["cvegeo_child"].dtype) == "string"
        # zero-padding must survive the CSV round-trip
        assert loaded["cvegeo_child"].str.len().eq(5).all()

    def test_seeding_does_not_clobber_an_edited_file(self, base_config, tmp_path):
        """
        The crosswalk is the one file in raw/ a human is expected to edit. A
        re-run must not overwrite their verification work.
        """
        geo.seed_crosswalk_if_absent(base_config)
        path = tmp_path / "raw" / "crosswalk.csv"
        edited = pd.read_csv(path, dtype={"cvegeo_child": "string",
                                          "cvegeo_parent": "string"})
        edited.loc[0, "source_note"] = "VERIFIED against INEGI catalogo by hand"
        edited.to_csv(path, index=False)

        geo.seed_crosswalk_if_absent(base_config)
        after = pd.read_csv(path)
        assert "VERIFIED against INEGI catalogo by hand" in after.loc[0, "source_note"]


class TestCvegeo:
    """
    Geographic codes are strings. These tests exist because the failure mode --
    an integer cast eating a leading zero -- is silent and produces a smaller
    but entirely plausible panel.
    """

    @pytest.mark.parametrize("ent,mun,expected", [
        (1, 1, "01001"),          # Aguascalientes: the leading-zero case
        ("1", "1", "01001"),
        ("01", "001", "01001"),
        (9, 17, "09017"),         # Mexico City
        (31, 50, "31050"),        # Yucatan
        (32, 58, "32058"),        # Zacatecas, highest state code
        ("1.0", "1.0", "01001"),  # survived a careless float round-trip
    ])
    def test_scalar_padding(self, ent, mun, expected):
        assert cvegeo(ent, mun) == expected

    @pytest.mark.parametrize("ent,mun", [(None, 1), (1, None), ("", "1"),
                                         ("abc", "1"), (float("nan"), 1)])
    def test_missing_or_invalid_gives_na(self, ent, mun):
        assert pd.isna(cvegeo(ent, mun))

    def test_overwide_code_raises_rather_than_truncating(self):
        """Truncating a too-wide code would corrupt the key silently."""
        with pytest.raises(ValueError, match="wider than"):
            cvegeo(123, 1)

    def test_series_padding(self):
        out = cvegeo_series(pd.Series(["1", "9", "31"]), pd.Series(["1", "17", "50"]))
        assert out.tolist() == ["01001", "09017", "31050"]
        assert str(out.dtype) == "string"

    def test_series_from_integers_still_pads(self):
        """The classic bug: a CSV read that inferred int64 for the state code."""
        out = cvegeo_series(pd.Series([1, 9, 31]), pd.Series([1, 17, 50]))
        assert out.tolist() == ["01001", "09017", "31050"]

    def test_series_handles_nulls(self):
        out = cvegeo_series(pd.Series(["1", None, "31"]), pd.Series(["1", "17", None]))
        assert out.iloc[0] == "01001"
        assert pd.isna(out.iloc[1])
        assert pd.isna(out.iloc[2])

    def test_series_rejects_overwide_codes(self):
        with pytest.raises(ValueError, match="wider than"):
            cvegeo_series(pd.Series(["123"]), pd.Series(["1"]))
