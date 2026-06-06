"""
exporters/json_exporter.py
Saves Python objects as pretty-printed UTF-8 JSON files.
"""

import json
from pathlib import Path


class JSONExporter:

    @staticmethod
    def save(data: object, filepath: str | Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)