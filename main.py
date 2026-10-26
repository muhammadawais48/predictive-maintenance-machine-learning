"""Run CSV ingestion, SQL feature extraction and model evaluation."""
from pathlib import Path
from database_setup import setup_database
from model_training import train_and_evaluate

def main():
    root = Path(__file__).resolve().parent
    rows = setup_database(root / "dataset.csv", root / "predictive_maintenance.db")
    print(f"Imported {rows:,} rows into SQLite")
    return train_and_evaluate(root / "predictive_maintenance.db", root / "outputs")

if __name__ == "__main__":
    main()
