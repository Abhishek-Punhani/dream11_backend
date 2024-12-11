from flask import Blueprint
from controllers.index import user_data , ai_alert,chatbot,generate_team_names,get_fantasy_points
from routes.final_script import process_teams



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

@user_blueprint.route("/names",methods=["GET"])
def get_names():
    return generate_team_names()


@user_blueprint.route('/get_fantasy_points', methods=['POST'])
def get_fantasy_points_route():
    return get_fantasy_points()


@user_blueprint.route('/get_team_data',methods=['POST'])
def get_team_data():
    return process_teams()