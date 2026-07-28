"""
Test fixtures.

The pipeline steps are named `03_distance.py` etc. -- leading digits make the
DAG order obvious in a directory listing, but they are not importable module
names. `load_step()` loads them by path so their functions can be unit-tested
directly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


def load_step(stem: str):
    """Import a numbered pipeline step by filename stem, e.g. '03_distance'."""
    path = SRC / f"{stem}.py"
    if not path.exists():
        pytest.skip(f"{path} not found")
    spec = importlib.util.spec_from_file_location(f"step_{stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def distance_mod():
    return load_step("03_distance")


@pytest.fixture(scope="session")
def assemble_mod():
    return load_step("07_assemble")


@pytest.fixture
def base_config(tmp_path):
    """A minimal config dict pointing at a temp directory."""
    return {
        "project": {"census_year": 2020, "migration_window_years": 5},
        "paths": {"raw": str(tmp_path / "raw"), "interim": str(tmp_path / "interim"),
                  "processed": str(tmp_path / "processed"),
                  "logs": str(tmp_path / "logs"), "reports": str(tmp_path / "reports")},
        "geometry": {"crosswalk_strategy": "aggregate_to_parent",
                     "crosswalk_file": str(tmp_path / "raw" / "crosswalk.csv")},
        "population": {"origin_pop_year": 2015, "dest_pop_year": 2020},
        "assemble": {"include_diagonal": False, "include_zero_flows": True,
                     "export_positive_only": False,
                     "drop_zero_flows_from_main_panel": False},
        "logging": {"level": "WARNING"},
    }
