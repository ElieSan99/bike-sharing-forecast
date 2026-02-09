import unittest
import pandas as pd
from src.data_loader import load_and_prepare_data, create_temporal_split
from pathlib import Path

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Création d'un petit dataframe de test
        self.test_df = pd.DataFrame({
            'datetime': pd.date_range(start='2024-12-31', periods=10, freq='h'),
            'demand': range(10)
        })
        self.test_csv = Path('tests/test_data.csv')
        self.test_df.to_csv(self.test_csv, index=False)

    def tearDown(self):
        if self.test_csv.exists():
            self.test_csv.unlink()

    def test_load_and_prepare_data(self):
        df = load_and_prepare_data(self.test_csv)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['datetime']))
        self.assertEqual(len(df), 10)

    def test_create_temporal_split(self):
        train, val, test = create_temporal_split(self.test_df, train_end='2024-12-31 02:00:00', val_end='2024-12-31 05:00:00')
        # On a 10h le 31 déc (00:00 à 09:00)
        # train: 00:00, 01:00, 02:00 (3h)
        # val: 03:00, 04:00, 05:00 (3h)
        # test: 06:00, 07:00, 08:00, 09:00 (4h)
        self.assertEqual(len(train), 3)
        self.assertEqual(len(val), 3)
        self.assertEqual(len(test), 4)


if __name__ == "__main__":
    unittest.main()

