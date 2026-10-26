"""Small reproducibility and data-integrity checks."""
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from database_setup import setup_database
from feature_engineering import extract_features
from model_training import train_and_evaluate

ROOT = Path(__file__).resolve().parents[1]

class PipelineTests(unittest.TestCase):
    def test_sql_import_and_features(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "test.db"
            self.assertEqual(setup_database(ROOT / "dataset.csv", db), 10000)
            df = extract_features(db)
            self.assertEqual(len(df), 10000)
            self.assertEqual(int(df["Machine_failure"].sum()), 589)
            self.assertFalse(df.isna().any().any())

    def test_training_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "test.db"
            setup_database(ROOT / "dataset.csv", db)
            metrics = train_and_evaluate(db, Path(temp) / "outputs")
            self.assertEqual(metrics["test_rows"], 2000)
            self.assertTrue(0 <= metrics["average_precision"] <= 1)
            self.assertTrue((Path(temp) / "outputs" / "metrics.json").exists())

if __name__ == "__main__":
    unittest.main()
