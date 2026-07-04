# FloodGuard — Flood Prediction System

🌐 **Live Demo:** [https://floodguard-ytqr.onrender.com/](https://floodguard-ytqr.onrender.com/)

A production-quality flood prediction web application built with Flask and scikit-learn/XGBoost. The system trains and compares multiple ML classifiers on historical flood data, serves the best model through a REST API, and displays predictions through a modern responsive web interface.


Everything below is a clean rebuild — the ML approach (train several classifiers, compare, keep the best, serve it through Flask) is preserved and improved.

---

## Project Structure

```
flood-prediction-system-main/
├── main.py
├── train.py
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── models.py
│   ├── ml.py
│   ├── routes.py
│   └── seed.py
├── templates/
│   ├── base.html
│   ├── home.html
│   ├── predict.html
│   ├── history.html
│   ├── 404.html
│   └── 500.html
├── static/
│   ├── css/style.css
│   ├── js/script.js
│   └── img/
│       ├── class_distribution.png
│       ├── correlation_heatmap.png
│       ├── model_comparison.png
│       └── best_model_confusion_matrix.png
├── models/
│   ├── flood_model.joblib
│   ├── scaler.joblib
│   └── metadata.json
└── data/
    └── flood_dataset.xlsx
```

---

## Dataset

`data/flood_dataset.xlsx` — 115 records with the following columns:

| Column | Meaning |
|---|---|
| `Temp` | Average temperature (°C) |
| `Humidity` | Relative humidity (%) |
| `Cloud Cover` | Cloud cover (%) |
| `ANNUAL` | Annual rainfall (mm) |
| `Jan-Feb` | Seasonal rainfall total — Jan to Feb (mm) |
| `Mar-May` | Seasonal rainfall total — Mar to May (mm) |
| `Jun-Sep` | Seasonal rainfall total — Jun to Sep (mm) |
| `Oct-Dec` | Seasonal rainfall total — Oct to Dec (mm) |
| `avgjune` | Average June rainfall (mm) |
| `sub` | Subdivision rainfall index |
| `flood` | **Target**: 1 = flood occurred, 0 = no flood |

The dataset is imbalanced (99 "no flood" vs 16 "flood" rows), which is why the training pipeline uses `class_weight="balanced"` / `scale_pos_weight` and selects the best model by **F1-score** rather than raw accuracy.

---

## Setup

Requires **Python 3.12**.

```bash
cd flood-prediction-system-main
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 1 — Train the Model

```bash
python train.py
```

This runs an 8-step pipeline:

1. Loads `data/flood_dataset.xlsx` and validates that all expected columns are present
2. Saves EDA charts (class balance bar chart, feature correlation heatmap) to `static/img/`
3. Cleans the data — median imputation for missing values, IQR-based outlier clipping
4. Splits train/test (80/20, stratified by flood class) and fits a `StandardScaler`
5. Trains four candidate classifiers: **Decision Tree**, **Random Forest**, **K-Nearest Neighbors**, **XGBoost**
6. Prints Accuracy / Precision / Recall / F1 for every model on the held-out test set
7. Picks the best model (Random Forest is preferred when its F1 is within 5 points of the top score, because it produces smoother, more realistic probabilities than a single Decision Tree)
8. Saves `flood_model.joblib`, `scaler.joblib`, and `metadata.json` to `models/`; also saves model-comparison and confusion-matrix charts to `static/img/`

**Results on the current dataset** (from `models/metadata.json`):

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Decision Tree | 95.65% | 100.00% | 66.67% | 80.00% |
| Random Forest | 95.65% | 100.00% | 66.67% | 80.00% |
| K-Nearest Neighbors | 86.96% | 50.00% | 33.33% | 40.00% |
| XGBoost | 86.96% | 0.00% | 0.00% | 0.00% |

Random Forest and Decision Tree tie on F1, so **Random Forest is automatically selected** as the deployed model (smoother probability output for the UI).

---

## 2 — Run the Web App

```bash
python main.py
```

Visit **http://localhost:5000**

### Pages

| Route | Description |
|---|---|
| `/` | Landing page — live model stats (accuracy, total predictions, flood alerts) and all training charts |
| `/predict` | Prediction form — enter weather data and get an instant flood/no-flood result with probability breakdown. Results render dynamically below the form via AJAX (no page reload). |
| `/history` | Table of the last 50 logged predictions, sorted newest first |

### REST API

The prediction form POSTs to an internal endpoint:

```
POST /api/predict
Content-Type: application/json

{
  "Temp": 28.5,
  "Humidity": 82,
  "Cloud Cover": 70,
  "ANNUAL": 1200,
  "Jan-Feb": 80,
  "Mar-May": 210,
  "Jun-Sep": 750,
  "Oct-Dec": 160,
  "avgjune": 180,
  "sub": 3.2
}
```

Response:
```json
{
  "prediction": 1,
  "flood_probability": 76.42,
  "no_flood_probability": 23.58,
  "message": "High flood risk detected. Consider issuing an alert.",
  "risk_level": "high",
  "model_name": "Random Forest",
  "saved": true
}
```

`risk_level` is `"high"` when flood probability ≥ 75%, `"moderate"` for lower flood predictions, and `"low"` when no flood is predicted.

---

## Database

Every prediction is persisted to a local SQLite database at `instance/flood_system.db` across four tables that mirror the project ERD:

| Table | Key Columns |
|---|---|
| `users` | UserID, Name, Email, Password, Role |
| `weather_data` | DataID, UserID, all 10 weather features, CreatedAt |
| `ml_models` | ModelID, ModelName, AlgorithmType, Accuracy, ModelFile, IsActive |
| `prediction_results` | PredictionID, DataID, ModelID, FloodResult, FloodProbability, PredictionDate |

A `guest` user (`guest@floodsystem.local`) is created automatically at startup — the schema supports adding real user accounts later without touching the prediction logic.

On first startup, **`seed.py`** runs 50 real dataset rows through the trained model and populates `prediction_results` with realistic timestamps spread over the past 10 days, so `/history` is never empty when you first open the app.

---

## Modules at a Glance

### `app/__init__.py` — App Factory
Creates the Flask app, initialises SQLAlchemy, creates all tables, ensures the guest user exists, syncs the `ml_models` table with `metadata.json`, and seeds the history on first run.

### `app/ml.py` — ML Helper
Loads `flood_model.joblib`, `scaler.joblib`, and `metadata.json` once at import time. Exposes `predict_flood(feature_dict)` which returns prediction, probabilities, a risk level, and a human-readable message. Raises `ModelNotReadyError` (a subclass of `RuntimeError`) if the model files are missing.

### `app/models.py` — Database Models
Defines all four SQLAlchemy ORM models. Also contains three helpers used at startup: `ensure_default_user()`, `sync_ml_model_registry()` (reads `metadata.json` and upserts the active model row), and `get_active_model_row()`.

### `app/routes.py` — HTTP Routes
Flask Blueprint with four routes: `GET /`, `GET /predict`, `GET /history`, and `POST /api/predict`. The API route validates all 10 required fields, calls `ml.predict_flood()`, and persists the result to the database (with a best-effort rollback on DB failure so the prediction response is still returned).

### `app/seed.py` — History Seeder
Reads up to 50 random rows from the dataset, runs them through the real trained model, and writes the results to `weather_data` and `prediction_results` with spread-out timestamps. Runs only once (skipped if `prediction_results` already has rows).

### `train.py` — Training Pipeline
Self-contained script. Loads the dataset, runs EDA, preprocesses, trains four classifiers, compares them, selects and saves the best one. All outputs (`.joblib` files, `metadata.json`, chart PNGs) are written to their respective directories. Re-run any time you want to retrain on new data.

---

## Re-training with New Data

Replace `data/flood_dataset.xlsx` (keep the same column names) and re-run:

```bash
python train.py
```

The next time you start `main.py`, the app factory calls `sync_ml_model_registry()` which automatically updates the `ml_models` table to reflect the newly trained model.

---

## Production Deployment

The built-in Flask dev server (`python main.py`) is for local development only. For production, run behind a WSGI server:

```bash
gunicorn "app:create_app()" --bind 0.0.0.0:8000 --workers 2
```

Set a real secret key via the `SECRET_KEY` environment variable before deploying.

---

## Notes

- If you see **"Model not available"** on the `/predict` page, `train.py` has not been run yet (or the `models/` directory was deleted). Run `python train.py` first.
- The `instance/` directory (SQLite DB) and `models/` directory are excluded from version control via `.gitignore`. Both are created automatically at runtime / training time.
- Python 3.12 is required. The `requirements.txt` pins all dependency versions for reproducibility.
