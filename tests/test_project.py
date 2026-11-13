import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from car_predictor import Car, DataProcessor, GraphPlotter, PricePredictor
import main

ROOT = Path(__file__).resolve().parents[1]


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.processor = DataProcessor(ROOT / 'car_data.csv')
        self.processor.load_data()
        self.data = self.processor.clean_data()
        self.predictor = PricePredictor()
        self.predictor.prepare_data(self.data)
        self.predictor.train()

    def test_predictions_and_metrics(self):
        for row in [(2025, 1200, 1.1), (2023, 1200, 1.2), (2001, 10000, 1.4)]:
            self.assertGreater(self.predictor.predict(*row), 0)
        metrics = self.predictor.evaluate()
        self.assertEqual(metrics['test_rows'], 1312)
        self.assertAlmostEqual(metrics['mae'], 167059.23, places=2)
        self.assertTrue(self.predictor.range_warnings(2025, 1200, 1.1))

    def test_invalid_and_out_of_range_inputs(self):
        for row in [(2018, -1, 1.5), (2018, 100, 0), (2018, float('nan'), 1.5),
                    (2018, 100, float('inf')), (2018.5, 100, 1.5)]:
            with self.subTest(row=row), self.assertRaises(ValueError):
                Car(*row)
        self.assertGreater(self.predictor.predict(2025, 0, 0.5), 0)
        self.assertGreater(self.predictor.predict(2018, 3000000, 4.0), 0)

    def test_bad_files_and_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.csv'
            with self.assertRaises(ValueError):
                DataProcessor(path).load_data()
            path.write_text('')
            with self.assertRaises(ValueError):
                DataProcessor(path).load_data()
            path.write_text('Year,Price\n2018,9000\n')
            with self.assertRaisesRegex(ValueError, 'Missing CSV columns'):
                DataProcessor(path).load_data()

    def test_cleaning_and_empty_dataset(self):
        bad = pd.DataFrame([[2018, -1, 1.5, 9000], ['bad', 40000, 1.5, 9000]], columns=self.data.columns)
        self.processor.data = pd.concat([self.data, self.data.iloc[:1], bad])
        self.assertEqual(len(self.processor.clean_data()), 6559)
        self.assertEqual(self.processor.removed_rows, 3)
        self.processor.data = bad
        with self.assertRaises(ValueError):
            self.processor.clean_data()

    def test_nonpositive_prediction(self):
        with patch.object(self.predictor.model, 'predict', return_value=[-10]):
            with self.assertRaisesRegex(ValueError, 'valid estimate'):
                self.predictor.predict(2018, 45000, 1.5)

    def test_logging_and_chart(self):
        import matplotlib
        matplotlib.use('Agg')
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with patch.object(main, 'OUTPUT_DIR', folder):
                for _ in range(2):
                    path = main.save_prediction(Car(2018, 45000, 1.5), 9327.96)
            rows = pd.read_csv(path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows.iloc[0]['EstimatedPriceINR'], 9327.96)
            graph = GraphPlotter(self.data).plot_all(folder / 'chart.png', show=False)
            self.assertGreater(graph.stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
