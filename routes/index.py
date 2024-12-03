from flask import Blueprint
from controllers.index import user_data , ai_alert,chatbot


user_blueprint = Blueprint("user", __name__)

@user_blueprint.route("/user_data", methods=["POST"])
def user_data_route():
    return user_data()

@user_blueprint.route("/ai_alert", methods=["POST"])
def ai_alert_route():
    return ai_alert()

@user_blueprint.route("/chat",methods=["POST"])
def chat():
    return chatbot()