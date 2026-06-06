"""
exporters/csv_exporter.py
Saves pandas DataFrames as UTF-8 CSV files.
"""

from pathlib import Path
import pandas as pd


class CSVExporter:

    @staticmethod
    def save(df: pd.DataFrame, filepath: str | Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False, encoding="utf-8")