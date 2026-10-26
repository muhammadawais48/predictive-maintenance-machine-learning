"""Import and validate the provided sensor dataset in a local SQLite database."""
from pathlib import Path
import sqlite3
import pandas as pd

REQUIRED_COLUMNS = {
    "UDI", "Product ID", "Type", "Air temperature [K]",
    "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]",
    "Tool wear [min]", "Machine failure"
}

def setup_database(csv_path="dataset.csv", db_path="predictive_maintenance.db"):
    csv_path, db_path = Path(csv_path), Path(db_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")
    if df.empty or df["UDI"].duplicated().any():
        raise ValueError("Dataset is empty or contains duplicate UDI values")
    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required fields contain missing values")
    if not df["Machine failure"].isin([0, 1]).all():
        raise ValueError("Machine failure must be binary (0 or 1)")
    df = df.rename(columns={
        "Product ID": "Product_ID", "Air temperature [K]": "Air_temperature_K",
        "Process temperature [K]": "Process_temperature_K",
        "Rotational speed [rpm]": "Rotational_speed_rpm",
        "Torque [Nm]": "Torque_Nm", "Tool wear [min]": "Tool_wear_min",
        "Machine failure": "Machine_failure"
    })
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql("sensor_data", conn, if_exists="replace", index=False)
    return len(df)

if __name__ == "__main__":
    print(f"Imported {setup_database():,} rows")
