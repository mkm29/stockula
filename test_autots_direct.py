#!/usr/bin/env python3
"""Test AutoTS directly to isolate the hanging issue."""

import pandas as pd
from autots import AutoTS
from datetime import datetime, timedelta


def test_autots_direct():
    """Test AutoTS predict directly."""
    print("Creating sample data...")

    # Create sample data
    dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
    data = pd.DataFrame(
        {
            "date": dates,
            "Close": 100
            + pd.Series(range(100))
            + pd.Series(range(100)).apply(lambda x: x * 0.1 * (-1 if x % 2 else 1)),
        }
    )

    print(f"Data shape: {data.shape}")
    print(data.head())

    print("\nCreating AutoTS model...")
    model = AutoTS(
        forecast_length=7,
        frequency="infer",
        prediction_interval=0.95,
        model_list=["LastValueNaive"],
        max_generations=1,
        num_validations=1,
        validation_method="backwards",
        verbose=1,
        n_jobs=1,
    )

    print("\nFitting model...")
    model = model.fit(
        data,
        date_col="date",
        value_col="Close",
        id_col=None,
    )

    print("\nModel fitted successfully!")
    print(f"Best model: {model.best_model_name}")

    print("\nCalling predict()...")
    prediction = model.predict()

    print("\nPredict completed!")
    print(f"Forecast shape: {prediction.forecast.shape}")
    print(prediction.forecast.head())


if __name__ == "__main__":
    test_autots_direct()
