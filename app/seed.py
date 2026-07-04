"""
seed.py
-------
Populates the history page with example predictions the first time the app
starts, so /history isn't empty before anyone has submitted a form. Uses
real rows from the dataset, run through the actual trained model — so the
probabilities shown (76%, 26%, etc.) are genuine model outputs, not
made-up numbers.
"""

import os
import random
from datetime import datetime, timedelta, timezone

import pandas as pd

from app import db, ml
from app.models import (
    PredictionResult,
    WeatherData,
    ensure_default_user,
    get_active_model_row,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "data", "flood_dataset.xlsx")


def seed_sample_predictions(count=50):
    # Don't duplicate — only seed if history is currently empty.
    if PredictionResult.query.count() > 0:
        return

    if not os.path.exists(DATASET_PATH):
        return

    model_row = get_active_model_row()
    if model_row is None:
        return

    df = pd.read_excel(DATASET_PATH)
    sample_df = df.sample(n=min(count, len(df)), random_state=7).reset_index(drop=True)

    user = ensure_default_user()
    now = datetime.now(timezone.utc)

    for _, row in sample_df.iterrows():
        features = {
            "Temp": float(row["Temp"]),
            "Humidity": float(row["Humidity"]),
            "Cloud Cover": float(row["Cloud Cover"]),
            "ANNUAL": float(row["ANNUAL"]),
            "Jan-Feb": float(row["Jan-Feb"]),
            "Mar-May": float(row["Mar-May"]),
            "Jun-Sep": float(row["Jun-Sep"]),
            "Oct-Dec": float(row["Oct-Dec"]),
            "avgjune": float(row["avgjune"]),
            "sub": float(row["sub"]),
        }

        try:
            result = ml.predict_flood(features)
        except Exception:
            continue

        # Spread timestamps over the past ~10 days so History looks realistic.
        timestamp = now - timedelta(
            hours=random.randint(1, 24 * 10), minutes=random.randint(0, 59)
        )

        weather_row = WeatherData(
            user_id=user.id,
            temp=features["Temp"],
            humidity=features["Humidity"],
            cloud_cover=features["Cloud Cover"],
            annual=features["ANNUAL"],
            jan_feb=features["Jan-Feb"],
            mar_may=features["Mar-May"],
            jun_sep=features["Jun-Sep"],
            oct_dec=features["Oct-Dec"],
            avgjune=features["avgjune"],
            sub=features["sub"],
            created_at=timestamp,
        )
        db.session.add(weather_row)
        db.session.flush()  # get weather_row.id before commit

        db.session.add(
            PredictionResult(
                data_id=weather_row.id,
                model_id=model_row.id,
                flood_result=result["prediction"],
                flood_probability=result["flood_probability"],
                prediction_date=timestamp,
            )
        )

    db.session.commit()
