import os
from flask import Flask
from flask_cors import CORS
from flask_session import Session
from routes.index import user_blueprint
from config import Config
from dotenv import load_dotenv


load_dotenv()

app = Flask(__name__)

print("CLIENT_URI:", os.getenv("CLIENT_URI"))

cors_options = {
    "supports_credentials": True,
    "origins": [f"{os.getenv('CLIENT_URI')}"],  # Your HTTP frontend
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
}
CORS(app, **cors_options)

# Load configuration
app.config.from_object(Config)

# Initialize session
Session(app)

# Register Blueprints
app.register_blueprint(user_blueprint, url_prefix="/api")

if __name__ == "__main__":
    app.run(debug=True)