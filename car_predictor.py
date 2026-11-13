# Author: Muhammad Usman Ghani
# Project: Used Car Price Estimator
# Course: Programming for AI

"""Data preparation, regression and charts for an educational car estimator."""
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = ['Year', 'Mileage', 'EngineSize']
COLUMNS = FEATURES + ['Price']


@dataclass
class Car:
    year: int
    mileage: float
    engine_size: float

    def __post_init__(self):
        values = [self.year, self.mileage, self.engine_size]
        if not all(np.isfinite(value) for value in values):
            raise ValueError('Please enter finite numbers, not NaN or infinity.')
        if self.year != int(self.year) or not 1886 <= self.year <= date.today().year:
            raise ValueError(f'Year must be a whole number from 1886 to {date.today().year}.')
        if self.mileage < 0:
            raise ValueError('Mileage cannot be negative.')
        if self.engine_size <= 0:
            raise ValueError('Engine size must be greater than zero.')

    def display(self):
        print(f'\nCar: {self.year} | {self.mileage:,.0f} km | {self.engine_size:g} L')


class DataProcessor:
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.data = None
        self.removed_rows = 0

    def load_data(self):
        try:
            self.data = pd.read_csv(self.filepath)
        except FileNotFoundError as exc:
            raise ValueError(f'Dataset not found: {self.filepath}') from exc
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            raise ValueError('Dataset is empty or is not a valid CSV file.') from exc
        missing = set(COLUMNS) - set(self.data.columns)
        if missing:
            raise ValueError('Missing CSV columns: ' + ', '.join(sorted(missing)))
        return self.data

    def clean_data(self):
        if self.data is None:
            raise ValueError('Load the dataset before cleaning it.')
        original_count = len(self.data)
        data = self.data[COLUMNS].apply(pd.to_numeric, errors='coerce')
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        valid = (
            data['Year'].between(1886, date.today().year)
            & (data['Year'] % 1 == 0)
            & (data['Mileage'] >= 0)
            & (data['EngineSize'] > 0)
            & (data['Price'] > 0)
        )
        self.data = data.loc[valid].drop_duplicates().reset_index(drop=True)
        self.removed_rows = original_count - len(self.data)
        if len(self.data) < 10:
            raise ValueError('At least 10 valid, unique rows are required for this demo.')
        return self.data

    def get_stats(self):
        return {
            'Total cars': len(self.data),
            'Rows removed': self.removed_rows,
            'Mean price (INR)': round(float(np.mean(self.data['Price'])), 2),
            'Minimum price (INR)': float(self.data['Price'].min()),
            'Maximum price (INR)': float(self.data['Price'].max()),
        }


class PricePredictor:
    def __init__(self):
        # Fit log(price) so inverse predictions are positive without clipping.
        self.model = TransformedTargetRegressor(
            regressor=LinearRegression(), func=np.log, inverse_func=np.exp
        )
        self.ranges = {}
        self.trained = False

    def prepare_data(self, data):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data[FEATURES], data['Price'], test_size=0.2, random_state=42
        )
        self.ranges = {
            column: (float(self.X_train[column].min()), float(self.X_train[column].max()))
            for column in FEATURES
        }
        self.trained = False

    def train(self):
        self.model.fit(self.X_train, self.y_train)
        self.trained = True

    def evaluate(self):
        predictions = self.model.predict(self.X_test)
        return {
            'mae': float(mean_absolute_error(self.y_test, predictions)),
            'r2': float(r2_score(self.y_test, predictions)),
            'train_rows': len(self.X_train),
            'test_rows': len(self.X_test),
        }

    def predict(self, year, mileage, engine_size):
        car = Car(year, mileage, engine_size)
        if not self.trained:
            raise ValueError('Train the model before requesting a prediction.')
        values = [car.year, car.mileage, car.engine_size]
        inputs = pd.DataFrame([values], columns=FEATURES)
        price = float(self.model.predict(inputs)[0])
        if not np.isfinite(price) or price <= 0:
            raise ValueError('The model cannot provide a valid estimate for these inputs.')
        return round(price, 2)


    def range_warnings(self, year, mileage, engine_size):
        """Explain extrapolation without blocking otherwise valid inputs."""
        warnings = []
        for name, value in zip(FEATURES, [year, mileage, engine_size]):
            low, high = self.ranges[name]
            if not low <= value <= high:
                warnings.append(f'{name} {value:g} is outside training data ({low:g}–{high:g}).')
        return warnings


class GraphPlotter:
    def __init__(self, data):
        self.data = data

    def plot_all(self, output_path, show=True):
        # Import only when charts are requested; predictions do not need a window.
        import matplotlib.pyplot as plt

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        labels = ['Manufacturing year', 'Mileage (km)', 'Engine size (L)']
        for ax, column, label, color in zip(axes, FEATURES, labels, ['#2563eb', '#dc2626', '#059669']):
            ax.scatter(self.data[column], self.data['Price'], color=color, alpha=0.8)
            ax.set(xlabel=label, ylabel='Price (INR)', title=f'{column} vs Price')
            ax.grid(alpha=0.2)
        fig.suptitle('Used Car Price Estimator — Dataset Overview')
        fig.tight_layout()
        try:
            fig.savefig(output_path, dpi=150)
            if show:
                plt.show()
        finally:
            plt.close(fig)
        return output_path
