"""数据 I/O 与路径工具。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import CLEAN_NAME, DATA_DIR, MANIFEST_CSV, NOISY_NAME


def load_manifest() -> pd.DataFrame:
    df = pd.read_csv(MANIFEST_CSV)
    df["caseid"] = df["caseid"].astype(str)
    return df


def case_paths(caseid: str) -> tuple[Path, Path]:
    from src.config import RAW_DIR
    case_dir = RAW_DIR / caseid
    return case_dir / NOISY_NAME, case_dir / CLEAN_NAME


def available_caseids() -> list[str]:
    caseids: list[str] = []
    for row in load_manifest().itertuples(index=False):
        cid = str(row.caseid)
        noisy, clean = case_paths(cid)
        if noisy.exists() and clean.exists():
            caseids.append(cid)
    return sorted(caseids)


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as f:
        return json.load(f)
