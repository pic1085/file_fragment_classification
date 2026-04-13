"""
Centralized path config with local overrides.

Local overrides (not committed):
  DEV/local/paths.py
  DEV/local/paths.json
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import json
from typing import Dict, Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULTS: Dict[str, str] = {
    "DATA_DIR": str(ROOT / "data"),
    "DATA_FEAT_DIR": str(ROOT / "data_feat"),
    "OUTPUT_DIR": str(ROOT / "outputs"),
    "RESULT_DIR": str(ROOT / "result"),
    "MODEL_DIR": str(ROOT / "model"),
    "MODEL_RUNS_DIR": str(ROOT / "model" / "runs"),
    "N_RUNS_DIR": str(ROOT / "N"),
    "UNKNOWN_RUNS_DIR": str(ROOT / "unknown"),
}


def _load_local_py() -> Dict[str, Any]:
    local_py = ROOT / "local" / "paths.py"
    if not local_py.exists():
        return {}
    spec = importlib.util.spec_from_file_location("local_paths", local_py)
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "PATHS") and isinstance(module.PATHS, dict):
        return module.PATHS
    # Fallback: collect UPPERCASE vars
    out: Dict[str, Any] = {}
    for k, v in module.__dict__.items():
        if k.isupper():
            out[k] = v
    return out


def _load_local_json() -> Dict[str, Any]:
    local_json = ROOT / "local" / "paths.json"
    if not local_json.exists():
        return {}
    try:
        with open(local_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _merge_paths(*sources: Dict[str, Any]) -> Dict[str, str]:
    merged: Dict[str, str] = dict(DEFAULTS)
    for src in sources:
        for k, v in src.items():
            if v is None:
                continue
            if isinstance(v, (str, Path)) and str(v).strip():
                merged[k] = str(Path(v).expanduser())
    return merged


PATHS = _merge_paths(_load_local_json(), _load_local_py())

# Convenience Path objects
DATA_DIR = Path(PATHS["DATA_DIR"])
DATA_FEAT_DIR = Path(PATHS["DATA_FEAT_DIR"])
OUTPUT_DIR = Path(PATHS["OUTPUT_DIR"])
RESULT_DIR = Path(PATHS["RESULT_DIR"])
MODEL_DIR = Path(PATHS["MODEL_DIR"])
MODEL_RUNS_DIR = Path(PATHS["MODEL_RUNS_DIR"])
N_RUNS_DIR = Path(PATHS["N_RUNS_DIR"])
UNKNOWN_RUNS_DIR = Path(PATHS["UNKNOWN_RUNS_DIR"])


def p(key: str, *parts: str) -> Path:
    return Path(PATHS[key]).joinpath(*parts)
