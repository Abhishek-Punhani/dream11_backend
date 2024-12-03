from flask import jsonify, request
import pandas as pd
import numpy as np
from prod_features.model.predict_model import predict_model
from prod_features.functionalities.features import (
    batting_strike_rate, bowling_economy, pitch_score, floor, ceil,
    batting_first_fantasy_points, chasing_first_fantasy_points,
    relative_points, matchup_rank, six_match_prediction,
     team_spider_chart
)
from utils.chatbot.chatbot import Chatbot
from utils.names import generate_team_names
from prod_features.functionalities.ai import CricInfoScraper, NewsScraper, process_multiple_headlines
# from prod_features.functionalities.ai import ai_alert

def user_data():
    try:
        data = request.get_json()
        player_id = data.get("player_id")
        match_no = data.get("match_no")
        # file_id = "1AuHiJyQF5P2YNgECfMykHaCVOVdUR32z"
        # url = f"https://drive.google.com/uc?id={file_id}"
        url=f"./prod_features/data/file_{match_no}.csv"
        df = pd.read_csv(url)
        df_copy = df.copy()
        df = df[df['player_id'] == player_id]
        
        # Get all values
        (y_pred_sorted, dream_team_points) = predict_model(df_copy, 'match_predictions', '')
        team1 = y_pred_sorted[:11]
        team2 = y_pred_sorted[11:]
        temp = float(dream_team_points) if isinstance(dream_team_points, (np.floating, np.integer)) else dream_team_points
        strike_rate = batting_strike_rate(df)
        economy = bowling_economy(df)
        score = float(pitch_score(player_id, df['venue'].iloc[0]))
        floor_value = float(floor(player_id))
        ceil_value = float(ceil(player_id))
        (batting_first_original_score, batting_first_predicted_score) = batting_first_fantasy_points(player_id, df)
        (chasing_first_original_score, chasing_first_predicted_score) = chasing_first_fantasy_points(player_id, df)
        points = relative_points(df_copy, player_id)
        rank = matchup_rank(df_copy, player_id)
        (y_actual, y_pred, date_of_match) = six_match_prediction(df)
        pfpp = float(dream_team_points / 11)  # Convert numpy.float64 to Python float
        (fes, doi, pcb) = team_spider_chart(df_copy, team1, dream_team_points)
        
        # Convert numpy numeric types to Python native types
        fes = int(fes) if isinstance(fes, np.integer) else fes
        doi = float(doi) if isinstance(doi, np.floating) else doi
        pcb = float(pcb) if isinstance(pcb, np.floating) else pcb

        response = {
            "strike_rate": strike_rate.tolist() if isinstance(strike_rate, (np.ndarray, pd.Series)) else float(strike_rate),
            "economy": economy.tolist() if isinstance(economy, (np.ndarray, pd.Series)) else float(economy),
            "score": score,
            "floor_value": floor_value,
            "ceil_value": ceil_value,
            "batting_first_original_score": batting_first_original_score.tolist() if isinstance(batting_first_original_score, (np.ndarray, pd.Series)) else float(batting_first_original_score),
            "batting_first_predicted_score": batting_first_predicted_score.tolist() if isinstance(batting_first_predicted_score, (np.ndarray, pd.Series)) else float(batting_first_predicted_score),
            "chasing_first_original_score": chasing_first_original_score.tolist() if isinstance(chasing_first_original_score, (np.ndarray, pd.Series)) else float(chasing_first_original_score),
            "chasing_first_predicted_score": chasing_first_predicted_score.tolist() if isinstance(chasing_first_predicted_score, (np.ndarray, pd.Series)) else float(chasing_first_predicted_score),
            "points": points.tolist() if isinstance(points, (np.ndarray, pd.Series)) else float(points),
            "rank": rank.tolist() if isinstance(rank, (np.ndarray, pd.Series)) else float(rank),
            "y_actual": y_actual.tolist() if isinstance(y_actual, (np.ndarray, pd.Series)) else y_actual,
            "y_pred": y_pred.tolist() if isinstance(y_pred, (np.ndarray, pd.Series)) else y_pred,
            "date_of_match": [str(d) for d in date_of_match] if isinstance(date_of_match, (list, np.ndarray, pd.Series)) else str(date_of_match),
            "team1": team1.to_dict('records') if isinstance(team1, pd.DataFrame) else [{'player_id': int(t[0]), 'y': float(t[1])} for t in team1] if isinstance(team1, np.ndarray) else team1,
            "team2": team2.to_dict('records') if isinstance(team2, pd.DataFrame) else [{'player_id': int(t[0]), 'y': float(t[1])} for t in team2] if isinstance(team2, np.ndarray) else team2,
            "temp": temp,
            "pfpp": pfpp,
            "fes": fes,
            "doi": doi,
            "pcb": pcb
        }

       
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        return jsonify({"error": str(e)}), 400

def ai_alert():
    try:
        data = request.get_json()
        player_id = data.get("player_id")
        match_no = data.get("match_no")

        # Read the CSV file based on match_no
        file_path = f"./prod_features/data/file_{match_no}.csv"
        df = pd.read_csv(file_path)

        # Extract the player name using player_id
        player_row = df[df['player_id'] == player_id]
        if player_row.empty:
            return jsonify({"error": "Player ID not found in the dataset."}), 404

        player_name = player_row['player'].iloc[0]
        print(player_name)

        # Read the set.csv file to get key_cricinfo
        set_file_path = "./prod_features/data/set.csv"
        set_df = pd.read_csv(set_file_path)

        # Extract key_cricinfo using player_id
        set_row = set_df[set_df['player_id'] == player_id]
        if set_row.empty:
            return jsonify({"error": "Player ID not found in set.csv."}), 404

        key_cricinfo = int(set_row['key_cricinfo'].iloc[0])
        print(f"Using key_cricinfo: {key_cricinfo}")

        # Step 1: Fetch URL using key_cricinfo
        news_url = CricInfoScraper.fetch_url(key_cricinfo)
        if news_url:
            print(f"News URL: {news_url}")

            # Step 2: Fetch Parsed HTML Content
            parsed_html = CricInfoScraper.fetch_response(news_url)
            if parsed_html:
                # Step 3: Extract News Content
                print("Fetching Latest News...")
                headlines = NewsScraper.fetch_content(parsed_html, max_headlines=5)

                if headlines:
                    # Step 4: Process Headlines with BERT
                    print("Processing Headlines...")
                    result = process_multiple_headlines(headlines, player_name)

                    # Output Results
                    response = {
                        "final_rating": result["final_rating"],
                        "insights": result["insights"]
                    }
                    return jsonify(response), 200
                else:
                    return jsonify({"error": "No headlines found."}), 404
            else:
                return jsonify({"error": "Failed to fetch parsed HTML content."}), 500
        else:
            return jsonify({"error": "Failed to fetch news URL."}), 500
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    

def chatbot():
     try:
        data = request.get_json()
        player1_id = data.get("player1_id")
        player2_id = data.get("player2_id")
        user_query = data.get("user_query")
        match_no = data.get("match_no")

        if match_no:
            chatbot = Chatbot(match_no)
        else:
            chatbot = Chatbot()

        response = chatbot.handle_query(player1_id, player2_id, user_query)

        return jsonify(response), 200
     except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
def generate_team_names_route():
    try:
        team_names = generate_team_names()
        return jsonify({"team_names": team_names}), 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500