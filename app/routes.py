"""
routes.py
---------
All HTTP routes for the Flood Prediction System, organized as a Blueprint.

Pages:
    GET  /            Home / landing page (project overview + model stats)
    GET  /predict      Prediction form page (results render dynamically below
                        the form via AJAX - no page reload needed)
    GET  /history       Table of previously logged predictions
    POST /api/predict   JSON API used by the prediction form's JavaScript
"""

from flask import Blueprint, jsonify, render_template, request

from app import db, ml
from app.models import (
    MLModel,
    PredictionResult,
    WeatherData,
    ensure_default_user,
    get_active_model_row,
)

main_bp = Blueprint("main", __name__)

# Keys the frontend form must send, and the human-readable labels used
# for validation error messages.
REQUIRED_FIELDS = {
    "Temp": "Temperature",
    "Humidity": "Humidity",
    "Cloud Cover": "Cloud Cover",
    "ANNUAL": "Annual Rainfall",
    "Jan-Feb": "Jan-Feb Rainfall",
    "Mar-May": "Mar-May Rainfall",
    "Jun-Sep": "Jun-Sep Rainfall",
    "Oct-Dec": "Oct-Dec Rainfall",
    "avgjune": "Average June Rainfall",
    "sub": "Subdivision Rainfall Index",
}


@main_bp.route("/")
def home():
    try:
        metadata = ml.get_metadata()
    except ml.ModelNotReadyError:
        metadata = None

    total_predictions = PredictionResult.query.count()
    flood_alerts = PredictionResult.query.filter_by(flood_result=1).count()

    return render_template(
        "home.html",
        metadata=metadata,
        total_predictions=total_predictions,
        flood_alerts=flood_alerts,
    )


@main_bp.route("/predict")
def predict_page():
    model_ready = True
    model_error = None
    try:
        ml.get_metadata()
    except ml.ModelNotReadyError as exc:
        model_ready = False
        model_error = str(exc)

    return render_template(
        "predict.html", model_ready=model_ready, model_error=model_error
    )


@main_bp.route("/history")
def history():
    records = (
        db.session.query(PredictionResult, WeatherData, MLModel)
        .join(WeatherData, PredictionResult.data_id == WeatherData.id)
        .join(MLModel, PredictionResult.model_id == MLModel.id)
        .order_by(PredictionResult.prediction_date.desc())
        .limit(50)
        .all()
    )
    return render_template("history.html", records=records)


@main_bp.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    missing = [label for key, label in REQUIRED_FIELDS.items() if key not in data or data[key] in (None, "")]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    try:
        numeric_data = {k: float(data[k]) for k in REQUIRED_FIELDS}
    except (TypeError, ValueError):
        return jsonify({"error": "All fields must be valid numbers."}), 400

    try:
        result = ml.predict_flood(numeric_data)
    except ml.ModelNotReadyError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive catch-all
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    # Persist to the database (User -> Weather_Data -> Prediction_Result)
    saved = False
    try:
        user = ensure_default_user()
        weather_row = WeatherData(
            user_id=user.id,
            temp=numeric_data["Temp"],
            humidity=numeric_data["Humidity"],
            cloud_cover=numeric_data["Cloud Cover"],
            annual=numeric_data["ANNUAL"],
            jan_feb=numeric_data["Jan-Feb"],
            mar_may=numeric_data["Mar-May"],
            jun_sep=numeric_data["Jun-Sep"],
            oct_dec=numeric_data["Oct-Dec"],
            avgjune=numeric_data["avgjune"],
            sub=numeric_data["sub"],
        )
        db.session.add(weather_row)
        db.session.flush()  # get weather_row.id before commit

        model_row = get_active_model_row()
        if model_row is not None:
            prediction_row = PredictionResult(
                data_id=weather_row.id,
                model_id=model_row.id,
                flood_result=result["prediction"],
                flood_probability=result["flood_probability"],
            )
            db.session.add(prediction_row)

        db.session.commit()
        saved = True
    except Exception:
        db.session.rollback()
        saved = False  # Prediction still succeeded; logging is best-effort

    result["saved"] = saved
    return jsonify(result), 200


@main_bp.errorhandler(404)
def not_found(_e):
    return render_template("404.html"), 404


@main_bp.errorhandler(500)
def server_error(_e):
    return render_template("500.html"), 500
