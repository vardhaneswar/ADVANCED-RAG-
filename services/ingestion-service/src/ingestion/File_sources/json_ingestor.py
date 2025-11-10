"""
JSON Ingestor (simple)
----------------------
Reads JSON dumps from a folder.
"""

from pathlib import Path
from typing import List, Dict
import json


def ingest_json(json_dir: Path) -> List[Dict]:
    """
    Read JSON files from a directory.

    Args:
        json_dir (Path): Directory containing *.json files.

    Returns:
        List[Dict]: Parsed JSON objects.
    """
    print(f"[json_ingestor] Looking for JSON in: {json_dir}")

    if not json_dir.exists():
        print(f"[json_ingestor] Directory does not exist: {json_dir}")
        return []

    items: List[Dict] = []
    json_files = list(json_dir.glob("*.json"))
    if not json_files:
        print("[json_ingestor] No JSON files found.")
        return []

    for path in json_files:
        print(f"[json_ingestor] Reading file: {path.name}")
        with path.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"[json_ingestor] Failed to parse: {path}")
                continue

            if isinstance(data, list):
                items.extend(data)
            else:
                items.append(data)

    print(f"[json_ingestor] Ingested {len(items)} JSON objects.")
    return items
