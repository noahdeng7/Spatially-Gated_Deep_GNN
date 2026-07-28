"""
Tests for the download integrity guards.

WHY THESE EXIST
---------------
During URL research (2026-07-21) INEGI was found to serve missing files as:

    HTTP 200 OK
    Content-Type: text/html
    1428 bytes
    "Esta liga ya no existe, lamentamos el inconveniente."

`raise_for_status()` does not catch that. Without a guard the downloader writes
an HTML error page to disk named `Censo2020_CA_nal_csv.zip`, `already_have()`
then reports the input as PRESENT on every subsequent run, and the failure
surfaces three steps later as an unintelligible parse error in 01_flows.

That is the exact class of silent corruption this pipeline is built to refuse,
so the guard gets a regression test.
"""

from __future__ import annotations

import pytest

from conftest import load_step

from common import PipelineError

download_mod = load_step("00_download")


class _Headers(dict):
    """requests uses case-insensitive headers; dict is close enough here."""
    def get(self, key, default=None):
        for k, v in self.items():
            if k.lower() == key.lower():
                return v
        return default


class TestContentTypeGuard:
    def test_html_where_zip_expected_raises(self, tmp_path):
        headers = _Headers({"content-type": "text/html; charset=utf-8",
                            "content-length": "1428"})
        with pytest.raises(PipelineError, match="HTML page"):
            download_mod._check_content_type(
                "https://www.inegi.org.mx/does/not/exist.zip",
                tmp_path / "exist.zip", headers, _log(),
            )

    def test_error_message_names_the_real_cause(self, tmp_path):
        """The message must point at the 200-with-error-page behaviour."""
        headers = _Headers({"content-type": "text/html", "content-length": "1428"})
        with pytest.raises(PipelineError) as exc:
            download_mod._check_content_type(
                "https://example.org/x.zip", tmp_path / "x.zip", headers, _log())
        msg = str(exc.value)
        assert "status 200" in msg
        assert "Nothing was written" in msg

    def test_html_where_tiff_expected_raises(self, tmp_path):
        headers = _Headers({"content-type": "text/html"})
        with pytest.raises(PipelineError, match="HTML page"):
            download_mod._check_content_type(
                "https://example.org/raster.tif", tmp_path / "raster.tif",
                headers, _log())

    def test_real_zip_passes(self, tmp_path):
        headers = _Headers({"content-type": "application/zip",
                            "content-length": "36615814"})
        download_mod._check_content_type(
            "https://example.org/iter.zip", tmp_path / "iter.zip", headers, _log())

    def test_real_tiff_passes(self, tmp_path):
        headers = _Headers({"content-type": "image/tiff"})
        download_mod._check_content_type(
            "https://example.org/pop.tif", tmp_path / "pop.tif", headers, _log())

    def test_octet_stream_passes(self, tmp_path):
        """Many servers send octet-stream for binaries; that must not fail."""
        headers = _Headers({"content-type": "application/octet-stream"})
        download_mod._check_content_type(
            "https://example.org/pop.tif", tmp_path / "pop.tif", headers, _log())

    def test_json_passes(self, tmp_path):
        headers = _Headers({"content-type": "application/json;charset=utf-8"})
        download_mod._check_content_type(
            "https://api.worldbank.org/x.json", tmp_path / "x.json", headers, _log())

    def test_html_is_allowed_when_html_was_requested(self, tmp_path):
        headers = _Headers({"content-type": "text/html"})
        download_mod._check_content_type(
            "https://example.org/page.html", tmp_path / "page.html", headers, _log())

    def test_unexpected_type_warns_but_does_not_raise(self, tmp_path):
        """A merely surprising type is logged, not fatal -- servers vary."""
        headers = _Headers({"content-type": "application/x-tar"})
        download_mod._check_content_type(
            "https://example.org/a.zip", tmp_path / "a.zip", headers, _log())

    def test_missing_content_type_does_not_raise(self, tmp_path):
        download_mod._check_content_type(
            "https://example.org/a.zip", tmp_path / "a.zip", _Headers({}), _log())


class TestManifestIntegrity:
    """Guards on the shipped config, so a careless edit is caught."""

    def test_every_entry_declares_a_confidence_tier(self):
        cfg = _config()
        for key, entry in (cfg.get("downloads") or {}).items():
            conf = (entry or {}).get("confidence")
            assert conf in {"VERIFIED", "INFERRED", "MANUAL"}, (
                f"downloads.{key} has confidence={conf!r}; must be one of "
                "VERIFIED / INFERRED / MANUAL so a reader knows how much to "
                "trust it"
            )

    def test_manual_entries_have_instructions(self):
        cfg = _config()
        for key, entry in (cfg.get("downloads") or {}).items():
            entry = entry or {}
            if entry.get("confidence") == "MANUAL":
                todo = (entry.get("todo") or "").strip()
                assert len(todo) > 40, (
                    f"downloads.{key} is MANUAL but has no usable instructions. "
                    "A human has to fetch it; tell them how."
                )

    def test_manual_entries_have_no_url(self):
        """MANUAL means no working URL -- a stale one would invite a bad fetch."""
        cfg = _config()
        for key, entry in (cfg.get("downloads") or {}).items():
            entry = entry or {}
            if entry.get("confidence") == "MANUAL":
                assert not entry.get("url"), (
                    f"downloads.{key} is MANUAL but still carries a url. "
                    "Remove it or re-tier the entry."
                )

    def test_entries_with_a_url_are_not_manual(self):
        cfg = _config()
        for key, entry in (cfg.get("downloads") or {}).items():
            entry = entry or {}
            if entry.get("url"):
                assert entry.get("confidence") in {"VERIFIED", "INFERRED"}

    def test_ppp_and_deflator_are_set(self):
        """04_gdp refuses to run without these; catch a null before the build."""
        ppp = _config()["gdp"]["ppp"]
        assert isinstance(ppp["conversion_factor"], (int, float))
        assert ppp["conversion_factor"] > 0
        assert isinstance(ppp["deflator_from_year_to_base"], (int, float))
        assert 0 < ppp["deflator_from_year_to_base"] <= 1.5


# --- helpers ---------------------------------------------------------------

def _log():
    import logging
    lg = logging.getLogger("test-download")
    lg.addHandler(logging.NullHandler())
    return lg


def _config():
    from common import load_config
    return load_config()
