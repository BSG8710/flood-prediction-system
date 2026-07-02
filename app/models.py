"""
Database models.

These map onto the entities from the project's ERD:

    User            (UserID, Name, Email, Password, Role)
    Weather_Data    (DataID, UserID, ...weather features..., created_at)
    ML_Model        (ModelID, ModelName, AlgorithmType, Accuracy, ModelFile)
    Prediction_Result (PredictionID, DataID, ModelID, FloodResult,
                        FloodProbability, PredictionDate)

Relationships:
    User          -> Weather_Data        (1:N)
    Weather_Data  -> Prediction_Result   (1:1)
    ML_Model      -> Prediction_Result   (1:N)

Note on Weather_Data columns: the original ERD used generic names
(AnnualRainfall, SeasonalRainfall, Temperature, Humidity, CloudVisibility).
The actual dataset supplied for this project (flood_dataset.xlsx) has a
richer, more specific set of weather features (monthly/seasonal rainfall
splits, etc). The table below is adapted to store exactly those columns so
every value used for a prediction is preserved, while keeping the same
overall entity/relationship shape as the ERD.
"""

import json
import os
from datetime import datetime, timezone

from werkzeug.security import generate_password_hash

from app import db

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA_PATH = os.path.join(BASE_DIR, "models", "metadata.json")

DEFAULT_USER_EMAIL = "guest@floodsystem.local"


class User(db.Model):
    __tablename__ = "users"

    id = db.Column("UserID", db.Integer, primary_key=True)
    name = db.Column("Name", db.String(120), nullable=False)
    email = db.Column("Email", db.String(160), unique=True, nullable=False)
    password_hash = db.Column("Password", db.String(255), nullable=False)
    role = db.Column("Role", db.String(20), default="guest", nullable=False)

    weather_records = db.relationship("WeatherData", back_populates="user")


class WeatherData(db.Model):
    __tablename__ = "weather_data"

    id = db.Column("DataID", db.Integer, primary_key=True)
    user_id = db.Column("UserID", db.Integer, db.ForeignKey("users.UserID"), nullable=False)

    # Feature columns - names mirror the dataset exactly for traceability.
    temp = db.Column("Temp", db.Float, nullable=False)
    humidity = db.Column("Humidity", db.Float, nullable=False)
    cloud_cover = db.Column("CloudCover", db.Float, nullable=False)
    annual = db.Column("Annual", db.Float, nullable=False)
    jan_feb = db.Column("JanFeb", db.Float, nullable=False)
    mar_may = db.Column("MarMay", db.Float, nullable=False)
    jun_sep = db.Column("JunSep", db.Float, nullable=False)
    oct_dec = db.Column("OctDec", db.Float, nullable=False)
    avgjune = db.Column("AvgJune", db.Float, nullable=False)
    sub = db.Column("Sub", db.Float, nullable=False)

    created_at = db.Column("CreatedAt", db.DateTime, default=lambda: datetime.now(timezone.utc))

    user = db.relationship("User", back_populates="weather_records")
    prediction = db.relationship(
        "PredictionResult", back_populates="weather_data", uselist=False
    )

    def to_feature_dict(self):
        """Return the feature values in the exact order the model expects."""
        return {
            "Temp": self.temp,
            "Humidity": self.humidity,
            "Cloud Cover": self.cloud_cover,
            "ANNUAL": self.annual,
            "Jan-Feb": self.jan_feb,
            "Mar-May": self.mar_may,
            "Jun-Sep": self.jun_sep,
            "Oct-Dec": self.oct_dec,
            "avgjune": self.avgjune,
            "sub": self.sub,
        }


class MLModel(db.Model):
    __tablename__ = "ml_models"

    id = db.Column("ModelID", db.Integer, primary_key=True)
    name = db.Column("ModelName", db.String(120), nullable=False)
    algorithm_type = db.Column("AlgorithmType", db.String(60), nullable=False)
    accuracy = db.Column("Accuracy", db.Float, nullable=False)
    model_file = db.Column("ModelFile", db.String(255), nullable=False)
    is_active = db.Column("IsActive", db.Boolean, default=True)

    predictions = db.relationship("PredictionResult", back_populates="ml_model")


class PredictionResult(db.Model):
    __tablename__ = "prediction_results"

    id = db.Column("PredictionID", db.Integer, primary_key=True)
    data_id = db.Column(
        "DataID", db.Integer, db.ForeignKey("weather_data.DataID"), nullable=False, unique=True
    )
    model_id = db.Column("ModelID", db.Integer, db.ForeignKey("ml_models.ModelID"), nullable=False)
    flood_result = db.Column("FloodResult", db.Integer, nullable=False)  # 0 or 1
    flood_probability = db.Column("FloodProbability", db.Float, nullable=False)
    prediction_date = db.Column(
        "PredictionDate", db.DateTime, default=lambda: datetime.now(timezone.utc)
    )

    weather_data = db.relationship("WeatherData", back_populates="prediction")
    ml_model = db.relationship("MLModel", back_populates="predictions")


# ----------------------------------------------------------------------
# Helpers used at startup
# ----------------------------------------------------------------------
def ensure_default_user():
    """Create a default 'guest' user so predictions can be logged without
    building a full authentication system (out of scope for this project,
    but the schema supports adding real accounts later)."""
    user = User.query.filter_by(email=DEFAULT_USER_EMAIL).first()
    if user is None:
        user = User(
            name="Guest",
            email=DEFAULT_USER_EMAIL,
            password_hash=generate_password_hash("guest-not-a-real-login"),
            role="guest",
        )
        db.session.add(user)
        db.session.commit()
    return user


def sync_ml_model_registry():
    """Read models/metadata.json (written by train.py) and make sure the
    ML_Model table has a matching, active row. Keeps the DB in sync with
    whatever model is actually on disk."""
    if not os.path.exists(METADATA_PATH):
        return None

    with open(METADATA_PATH) as f:
        meta = json.load(f)

    best_name = meta.get("best_model_name")
    accuracy = meta.get("metrics", {}).get(best_name, {}).get("accuracy", 0.0)

    existing = MLModel.query.filter_by(name=best_name, model_file="flood_model.joblib").first()
    if existing:
        existing.accuracy = accuracy
        existing.is_active = True
    else:
        # Deactivate old models, register the new best one as active.
        MLModel.query.update({MLModel.is_active: False})
        existing = MLModel(
            name=best_name,
            algorithm_type=best_name,
            accuracy=accuracy,
            model_file="flood_model.joblib",
            is_active=True,
        )
        db.session.add(existing)

    db.session.commit()
    return existing


def get_active_model_row():
    return MLModel.query.filter_by(is_active=True).order_by(MLModel.id.desc()).first()
