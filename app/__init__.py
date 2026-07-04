"""
Flask application factory.

Keeping app creation in a factory function (instead of a bare module-level
`app = Flask(__name__)`) makes the project testable and avoids circular
imports between routes, models, and the ML helper.
"""

import os

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")


def create_app():
    os.makedirs(INSTANCE_DIR, exist_ok=True)

    app = Flask(
        __name__,
        template_folder=os.path.join(BASE_DIR, "templates"),
        static_folder=os.path.join(BASE_DIR, "static"),
    )

    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(
        INSTANCE_DIR, "flood_system.db"
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)

    from app import models  # noqa: F401  (needed so tables are registered)

    with app.app_context():
        db.create_all()
        models.ensure_default_user()
        models.sync_ml_model_registry()

        from app import seed
        seed.seed_sample_predictions(50)

    from app.routes import main_bp

    app.register_blueprint(main_bp)

    return app    from app import models  # noqa: F401  (needed so tables are registered)

   with app.app_context():
        db.create_all()
        models.ensure_default_user()
        models.sync_ml_model_registry()

        from app import seed
        seed.seed_sample_predictions(50)

    from app.routes import main_bp

    app.register_blueprint(main_bp)

    return app
