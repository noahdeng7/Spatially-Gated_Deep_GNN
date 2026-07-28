"""
Shared helpers for the mx-migration pipeline.

Not one of the numbered steps: this module exists so that every step handles
geographic codes, logging and IO identically. The single most important thing in
here is `cvegeo()` -- geographic codes are strings, forever, and the moment one
becomes an integer the leading zero on Aguascalientes (01) disappears and joins
start failing silently.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import yaml

# Repo root = parent of src/
ROOT = Path(__file__).resolve().parent.parent

# The canonical string dtype for every geographic key in this pipeline.
CVEGEO_DTYPE = "string"

__all__ = [
    "ROOT",
    "load_config",
    "get_logger",
    "resolve",
    "ensure_dirs",
    "cvegeo",
    "cvegeo_series",
    "assert_cvegeo_valid",
    "read_interim",
    "write_interim",
    "rowlog",
    "log_rows",
    "missingness_table",
    "read_csv_smart",
    "report_path",
    "rel",
    "run_step",
    "PipelineError",
    "DecisionRequired",
]


class PipelineError(RuntimeError):
    """A validation guard failed. The build should stop."""


class DecisionRequired(PipelineError):
    """
    A material analytic choice cannot be made automatically.

    Raised (for example) when the two GDP sources disagree beyond the configured
    correlation floor. The pipeline stops and asks rather than silently picking.
    """


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml. Defaults to the one next to this repo's src/."""
    path = Path(path) if path else ROOT / "config.yaml"
    if not path.exists():
        raise PipelineError(f"config not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(path)
    return cfg


def resolve(cfg: Mapping[str, Any], relpath: str | Path) -> Path:
    """Resolve a config-relative path against the repo root."""
    p = Path(relpath)
    return p if p.is_absolute() else (ROOT / p)


def rel(path: str | Path) -> str:
    """
    Path rendered relative to the repo root, for log lines.

    Falls back to the absolute path when `path` lies outside the repo, which is
    legitimate -- config may point data or reports at an external drive. Plain
    `Path.relative_to(ROOT)` raises ValueError in that case, and crashing a log
    statement is a poor reason to lose a run.
    """
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def ensure_dirs(cfg: Mapping[str, Any]) -> None:
    """Create the standard data/log/report directories if absent."""
    for key in ("raw", "interim", "processed", "logs", "reports"):
        configured = cfg.get("paths", {}).get(key)
        if configured:
            resolve(cfg, configured).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------

_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str, cfg: Mapping[str, Any] | None = None) -> logging.Logger:
    """
    Logger that writes to stdout and appends to the shared build log.

    Every step uses the same build log so that `rows in / rows out` for the whole
    pipeline is one greppable file.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.propagate = False
    level = (cfg or {}).get("logging", {}).get("level", "INFO")
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-7s  %(name)-14s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if cfg:
        log_path = resolve(cfg, (cfg.get("logging", {}) or {}).get("build_log", "logs/build.log"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    _LOGGERS[name] = logger
    return logger


def log_rows(logger: logging.Logger, label: str, n_in: int, n_out: int) -> None:
    """Uniform 'rows in / rows out' line. Every step emits one per operation."""
    delta = n_out - n_in
    pct = (delta / n_in * 100.0) if n_in else float("nan")
    logger.info(
        "ROWS  %-38s  in=%-12s out=%-12s delta=%+d (%+.2f%%)",
        label, f"{n_in:,}", f"{n_out:,}", delta, pct,
    )


@contextmanager
def rowlog(logger: logging.Logger, label: str, df_in: Any):
    """
    Context manager that logs rows in / rows out around a transformation.

    Usage:
        with rowlog(log, "drop non-migrants", df) as r:
            df = df[df.orig != df.dest]
            r.out = df
    """
    class _R:
        out: Any = None

    n_in = _nrows(df_in)
    r = _R()
    t0 = time.perf_counter()
    yield r
    n_out = _nrows(r.out) if r.out is not None else n_in
    log_rows(logger, label, n_in, n_out)
    logger.debug("      %s took %.2fs", label, time.perf_counter() - t0)


def _nrows(obj: Any) -> int:
    if obj is None:
        return 0
    if hasattr(obj, "height"):        # polars
        return int(obj.height)
    if hasattr(obj, "shape"):         # pandas / geopandas / numpy
        return int(obj.shape[0])
    if isinstance(obj, int):
        return obj
    try:
        return len(obj)
    except TypeError:
        return 0


# ---------------------------------------------------------------------------
# CVEGEO: geographic codes are strings, always
# ---------------------------------------------------------------------------

def cvegeo(ent: Any, mun: Any) -> str | pd.NA:
    """
    Build a 5-character CVEGEO from a state and municipality code.

    2-digit zero-padded state + 3-digit zero-padded municipality.
    Accepts ints, floats-that-are-ints, or strings with or without padding.
    Returns pd.NA if either component is missing or non-numeric.

        cvegeo(1, 1)       -> "01001"
        cvegeo("1", "1")   -> "01001"
        cvegeo("01", "001")-> "01001"
        cvegeo(9, 17)      -> "09017"
    """
    e = _norm_code(ent, 2)
    m = _norm_code(mun, 3)
    if e is None or m is None:
        return pd.NA
    return e + m


def _norm_code(value: Any, width: int) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "<na>"}:
        return None
    # Tolerate values that arrived as floats ("1.0") from a careless CSV read.
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    if not s.isdigit():
        return None
    if len(s) > width:
        # Already wider than expected -- do not truncate, that would corrupt the
        # key. Surface it instead.
        raise ValueError(f"code {s!r} is wider than the expected {width} digits")
    return s.zfill(width)


def cvegeo_series(ent: pd.Series, mun: pd.Series) -> pd.Series:
    """
    Vectorised CVEGEO construction over two Series.

    Returns a pandas `string` Series (never object, never int).
    """
    e = _norm_series(ent, 2)
    m = _norm_series(mun, 3)
    out = e + m
    out = out.where(e.notna() & m.notna(), pd.NA)
    return out.astype(CVEGEO_DTYPE)


def _norm_series(s: pd.Series, width: int) -> pd.Series:
    t = s.astype(CVEGEO_DTYPE).str.strip()
    t = t.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    # strip a trailing ".0" left by a float round-trip
    t = t.str.replace(r"^(\d+)\.0$", r"\1", regex=True)
    t = t.where(t.str.fullmatch(r"\d+"), pd.NA)
    too_wide = t.notna() & (t.str.len() > width)
    if bool(too_wide.any()):
        bad = t[too_wide].unique()[:5].tolist()
        raise ValueError(
            f"{int(too_wide.sum())} code(s) wider than {width} digits, e.g. {bad}. "
            "Refusing to truncate a geographic key."
        )
    return t.str.zfill(width)


def assert_cvegeo_valid(s: pd.Series, name: str = "CVEGEO") -> None:
    """Hard guard: 5 digits, string dtype, no nulls."""
    if str(s.dtype) not in {"string", "object"}:
        raise PipelineError(
            f"{name} has dtype {s.dtype}; geographic codes must be strings. "
            "An integer cast has silently eaten leading zeros somewhere upstream."
        )
    bad_len = s.notna() & (s.astype("string").str.len() != 5)
    if bool(bad_len.any()):
        raise PipelineError(
            f"{name}: {int(bad_len.sum())} value(s) are not 5 characters, "
            f"e.g. {s[bad_len].unique()[:5].tolist()}"
        )
    n_null = int(s.isna().sum())
    if n_null:
        raise PipelineError(f"{name}: {n_null} null code(s)")


# ---------------------------------------------------------------------------
# interim IO
# ---------------------------------------------------------------------------

def _interim_path(cfg: Mapping[str, Any], name: str) -> Path:
    d = resolve(cfg, cfg["paths"]["interim"])
    d.mkdir(parents=True, exist_ok=True)
    return d / (name if name.endswith(".parquet") else f"{name}.parquet")


def write_interim(cfg: Mapping[str, Any], df: pd.DataFrame, name: str,
                  logger: logging.Logger | None = None) -> Path:
    """
    Write an interim table to parquet.

    Parquet (not CSV) because it round-trips the `string` dtype on CVEGEO
    columns. A CSV round-trip is exactly where a zero-padded code turns back
    into an integer.
    """
    path = _interim_path(cfg, name)
    for col in df.columns:
        if col.lower().startswith("cvegeo") or col in {"orig", "dest"}:
            df[col] = df[col].astype(CVEGEO_DTYPE)
    df.to_parquet(path, index=False, compression="zstd")
    if logger:
        logger.info("WROTE %-38s  rows=%-12s cols=%-4d -> %s",
                    name, f"{len(df):,}", df.shape[1], rel(path))
    return path


def read_interim(cfg: Mapping[str, Any], name: str,
                 logger: logging.Logger | None = None) -> pd.DataFrame:
    """Read an interim parquet, restoring string dtype on key columns."""
    path = _interim_path(cfg, name)
    if not path.exists():
        raise PipelineError(
            f"missing interim table: {rel(path)}. "
            "Run the step that produces it first (see Makefile for the DAG)."
        )
    df = pd.read_parquet(path)
    for col in df.columns:
        if col.lower().startswith("cvegeo") or col in {"orig", "dest"}:
            df[col] = df[col].astype(CVEGEO_DTYPE)
    if logger:
        logger.info("READ  %-38s  rows=%-12s cols=%d", name, f"{len(df):,}", df.shape[1])
    return df


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def missingness_table(df: pd.DataFrame, logger: logging.Logger | None = None) -> pd.DataFrame:
    """
    Per-column missingness summary, printed before anything is written.

    Returns the table so callers can also persist it to reports/.
    """
    n = len(df)
    rows = []
    for col in df.columns:
        n_null = int(df[col].isna().sum())
        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "n_missing": n_null,
            "pct_missing": round(100.0 * n_null / n, 4) if n else float("nan"),
        })
    out = pd.DataFrame(rows).sort_values("pct_missing", ascending=False, ignore_index=True)
    if logger:
        logger.info("MISSINGNESS (n=%s rows)", f"{n:,}")
        for _, r in out.iterrows():
            logger.info("      %-28s %-12s %12s  %7.3f%%",
                        r["column"], r["dtype"], f"{r['n_missing']:,}", r["pct_missing"])
    return out


def read_csv_smart(path: str | Path, logger: logging.Logger | None = None,
                   **kwargs) -> pd.DataFrame:
    """
    Read an INEGI CSV without guessing its encoding wrong.

    INEGI is not consistent. The ITER 2020 file is UTF-8 with a BOM; other
    products are genuinely latin-1. Reading UTF-8 as latin-1 does not raise --
    it silently mojibakes every accented character, so "Miacatlan" becomes
    "MiacatlAin" and a degree sign becomes two characters. That is harmless for
    joins keyed on CVEGEO, but the mangled names end up in reports and in the
    published codebook.

    Order: utf-8-sig (strips a BOM if present) -> utf-8 -> latin-1. latin-1
    cannot fail, so it is the backstop; the earlier attempts are what stop it
    from being used when the file is really UTF-8.
    """
    kwargs.pop("encoding", None)
    last: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc, **kwargs)
            if logger and enc != "utf-8-sig":
                logger.info("      read %s as %s", Path(path).name, enc)
            return df
        except UnicodeDecodeError as exc:
            last = exc
            continue
    raise PipelineError(f"could not decode {path}: {last}")


def report_path(cfg: Mapping[str, Any], name: str) -> Path:
    d = resolve(cfg, cfg["paths"]["reports"])
    d.mkdir(parents=True, exist_ok=True)
    return d / name


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------

def run_step(main_fn, step: str) -> int:
    """
    Run a step's main() and render pipeline failures as readable messages.

    A PipelineError means a guard fired -- a missing input, a failed validation,
    a decision that needs a human. Those messages are written to be read, and a
    40-line traceback around them buries the part that matters. Unexpected
    exceptions still get their full traceback, because for those the stack IS
    the information.

    Exit codes:  0 ok  |  1 guard failed  |  2 decision required  |  3 unexpected
    """
    try:
        return int(main_fn() or 0)
    except DecisionRequired as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        print(f"[{step}] stopped: this choice needs a human. Nothing was written.",
              file=sys.stderr)
        return 2
    except PipelineError as exc:
        print(f"\n[{step}] FAILED\n\n{exc}\n", file=sys.stderr)
        return 1
    except NotImplementedError as exc:
        print(f"\n[{step}] not implemented\n\n{exc}\n", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"\n[{step}] interrupted", file=sys.stderr)
        return 130


