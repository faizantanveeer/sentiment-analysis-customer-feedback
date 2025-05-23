from flask import Flask
from .routes import bp as routes_bp
from .db import close_db
from flask_cors import CORS


def create_app():
    app = Flask(__name__)

    CORS(app)

    # Register routes
    app.register_blueprint(routes_bp)

    # Register DB cleanup
    app.teardown_appcontext(close_db)

    return app
