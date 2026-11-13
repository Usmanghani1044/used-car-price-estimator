# Author: Muhammad Usman Ghani
# Project: Used Car Price Estimator
# Course: Programming for AI

"""Run with: python main.py"""
import csv
from datetime import datetime, timezone
from pathlib import Path

from car_predictor import Car, DataProcessor, GraphPlotter, PricePredictor

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'outputs'


def save_prediction(car, price):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / 'predictions_inr.csv'
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        if needs_header:
            writer.writerow(['TimestampUTC', 'Year', 'MileageKm', 'EngineSizeL', 'EstimatedPriceINR'])
        writer.writerow([datetime.now(timezone.utc).isoformat(timespec='seconds'),
                         car.year, car.mileage, car.engine_size, price])
    return path


def main():
    print('\n===== USED CAR PRICE ESTIMATOR =====')
    print('Educational demo | Prices in INR | Not a market valuation service')
    try:
        processor = DataProcessor(BASE_DIR / 'car_data.csv')
        processor.load_data()
        data = processor.clean_data()
        predictor = PricePredictor()
        predictor.prepare_data(data)
        predictor.train()
        metrics = predictor.evaluate()
    except (ValueError, OSError) as exc:
        print(f'Cannot start: {exc}')
        return 1

    print(f'Loaded {len(data)} cars; removed {processor.removed_rows} invalid/duplicate rows.')
    print(f'Test MAE: INR {metrics["mae"]:,.2f} | R²: {metrics["r2"]:.3f}')
    print(f'Split: {metrics["train_rows"]} training / {metrics["test_rows"]} test rows.')
    print('Historical Indian listings: results are not current market valuations.')

    while True:
        print('\n1. Estimate a price\n2. View and save graphs\n3. View dataset statistics\n4. Exit')
        choice = input('Choose 1–4: ').strip()
        if choice == '1':
            try:
                year = int(input('Manufacturing year: '))
                mileage = float(input('Mileage in km: '))
                engine_size = float(input('Engine size in L: '))
                car = Car(year, mileage, engine_size)
                price = predictor.predict(year, mileage, engine_size)
            except ValueError as exc:
                print(f'Cannot estimate: {exc}')
                continue
            for warning in predictor.range_warnings(year, mileage, engine_size):
                print(f'Warning: {warning} Extrapolated estimate; use caution.')
            car.display()
            print(f'Estimated price: INR {price:,.2f} INR')
            print('Demo estimate in Indian rupees (INR); not a verified market price.')
            try:
                print(f'Saved to: {save_prediction(car, price)}')
            except OSError as exc:
                print(f'Estimate calculated, but saving failed: {exc}')
        elif choice == '2':
            print('Close the graph window to return to this menu.')
            try:
                path = GraphPlotter(data).plot_all(OUTPUT_DIR / 'car_graphs.png')
                print(f'Graph saved to: {path}')
            except (OSError, RuntimeError, ImportError) as exc:
                print(f'Could not display or save graphs: {exc}')
        elif choice == '3':
            for label, value in processor.get_stats().items():
                print(f'{label}: {value:,}')
        elif choice == '4':
            print('Goodbye!')
            return 0
        else:
            print('Please choose 1, 2, 3, or 4.')


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print('\nSession ended.')
