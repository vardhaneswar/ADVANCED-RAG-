"""
CSV Ingestor (real)
-------------------
Reads CSV files with tabular financial data.
"""

from pathlib import Path
from typing import List, Dict

import pandas as pd


def ingest_csvs(csv_dir: Path) -> List[Dict]:
    """
    Read all CSV files in a directory into a list of row dicts.

    Args:
        csv_dir (Path): Directory containing CSV files.

    Returns:
        List[Dict]: All rows from all CSVs.
    """
    print(f"[csv_ingestor] Looking for CSVs in: {csv_dir}")

    if not csv_dir.exists():
        print(f"[csv_ingestor] Directory does not exist: {csv_dir}")
        return []

    all_rows: List[Dict] = []
    csv_files = list(csv_dir.glob("*.csv"))

    if not csv_files:
        print("[csv_ingestor] No CSV files found.")
        return []

    for path in csv_files:
        print(f"[csv_ingestor] Reading file: {path.name}")
        df = pd.read_csv(path)
        df["__source_file"] = path.name
        rows = df.to_dict(orient="records")
        all_rows.extend(rows)

    print(f"[csv_ingestor] Ingested {len(all_rows)} rows from {len(csv_files)} file(s).")
    return all_rows
