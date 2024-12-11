import os
import pandas as pd
import numpy as np
import json
import requests
from flask import jsonify, request

from prod_features.model.predict_model import predict_model
from prod_features.functionalities.features import (
    batting_strike_rate, bowling_economy, pitch_score, floor, ceil,
    batting_first_fantasy_points, chasing_first_fantasy_points,
    relative_points, matchup_rank, six_match_prediction,
    team_spider_chart, risk_assesment
)
from utils.chatbot.chatbot import Chatbot
from utils.names import generate_team_names
from prod_features.functionalities.ai import CricInfoScraper, NewsScraper, process_multiple_headlines

def process_teams():
    data = request.get_json()
    player_id = data.get("player_id")
    team1 = data.get("team1")
    team2 = data.get("team2")
    date = data.get("date")
    df = pd.read_csv(os.path.join("/home/manav/dev_ws/src/dream11_backend/prod_features/data", "full_dataset.csv"))
    df = df[(df['team'] == team1) | (df['team'] == team2)]
    df = df[df['start_date'] == date]

    df = df[['player', 'player_id', 'team', 'opponent', 'start_date', 'end_date', 'venue', 'match_type', 'is_captain', 'is_player_of_match', 'runs_scored', 'balls_faced', 'outs', 'fours', 'sixes', 'runs_conceded', 'balls_bowled', 'wickets', 'catches', 'stumpings', 'runouts', 'wickets_taken_players', 'dismissed_by', 'wicket_type', 'runs_powerplay', 'fours_powerplay', 'sixes_powerplay', 'wickets_powerplay', 'runs_middle', 'fours_middle', 'sixes_middle', 'wickets_middle', 'runs_death', 'fours_death', 'sixes_death', 'wickets_death', 'win', 'win_by_run', 'win_by_wickets', 'maidens', 'gender', 'batting_average', 'batting_strike_rate', 'bowling_average', 'bowling_strike_rate', 'bowling_economy']]
    print("Initial DataFrame structure:")
    print(df.head())

    if 'values' in df.columns:
        df.drop(columns=['values'], inplace=True)
    if 'ai_alerts' in df.columns:
        df.drop(columns=['ai_alerts'], inplace=True)

    df['values'] = np.nan
    df['ai_alerts'] = np.nan

    for index, row in df.iterrows():
        player_id = row['player_id']

        try:
            print(f"\nProcessing player {player_id}...")

            user_data_response = user_data(team1, team2, date, player_id)
            ai_alert_response = ai_alert(player_id, team1, team2, date)

            df.at[index, 'values'] = json.dumps(user_data_response)
            df.at[index, 'ai_alerts'] = json.dumps(ai_alert_response)
            print(f"Successfully processed player {player_id}")
        except Exception as e:
            print(f"Error processing player {player_id}: {str(e)}")
            continue

    print("Final DataFrame structure:")
    print(df.head())
    output_path = '/home/manav/dev_ws/src/dream11_backend/data/New/file_6.csv'
    df.to_csv(output_path, index=False)
    print(f"\nUpdated CSV saved to {output_path}")
    return jsonify({"message": "Data processed successfully!"})

def user_data(team1, team2, date, player_id):
    try:
        url = "./prod_features/data/full_dataset.csv"
        df = pd.read_csv(url)
        df = df[(df['team'] == team1) | (df['team'] == team2)]
        df = df[df['start_date'] == date]
        print("Initial DataFrame structure:")
        print(df.head())

        df = df[['player', 'player_id', 'team', 'opponent', 'start_date', 'end_date', 'venue', 'match_type', 'is_captain', 'is_player_of_match', 'runs_scored', 'balls_faced', 'outs', 'fours', 'sixes', 'runs_conceded', 'balls_bowled', 'wickets', 'catches', 'stumpings', 'runouts', 'wickets_taken_players', 'dismissed_by', 'wicket_type', 'runs_powerplay', 'fours_powerplay', 'sixes_powerplay', 'wickets_powerplay', 'runs_middle', 'fours_middle', 'sixes_middle', 'wickets_middle', 'runs_death', 'fours_death', 'sixes_death', 'wickets_death', 'win', 'win_by_run', 'win_by_wickets', 'maidens', 'gender', 'batting_average', 'batting_strike_rate', 'bowling_average', 'bowling_strike_rate', 'bowling_economy']]
       
        df_copy = df.copy()
        df_copy_risk = pd.read_csv(url)
        df = df[df['player_id'] == player_id]
        
        (y_pred_sorted, dream_team_points, mod_player_id) = predict_model(df_copy, 'match_predictions', '')
        print("Predicted sorted values:", y_pred_sorted)
        print("Dream team points:", dream_team_points)
        print("Mod player ID:", mod_player_id)
        
        team1 = y_pred_sorted[:11]
        team2 = y_pred_sorted[11:]
        temp = float(dream_team_points) if isinstance(dream_team_points, (np.floating, np.integer)) else dream_team_points
        strike_rate = batting_strike_rate(df)
        print("Strike rate:", strike_rate)
        economy = bowling_economy(df)
        print("Economy:", economy)
        score = float(pitch_score(player_id, df['venue'].iloc[0]))
        print("Score:", score)
        floor_value = float(floor(player_id))
        print("Floor value:", floor_value)
        ceil_value = float(ceil(player_id))
        print("Ceil value:", ceil_value)
        (batting_first_original_score, batting_first_predicted_score) = batting_first_fantasy_points(player_id, df)
        print("Batting first original score:", batting_first_original_score)
        print("Batting first predicted score:", batting_first_predicted_score)
        (chasing_first_original_score, chasing_first_predicted_score) = chasing_first_fantasy_points(player_id, df)
        print("Chasing first original score:", chasing_first_original_score)
        print("Chasing first predicted score:", chasing_first_predicted_score)
        points = relative_points(df_copy, player_id)
        print("Points:", points)
        rank = matchup_rank(df_copy, player_id)
        print("Rank:", rank)
        (y_actual, y_pred, date_of_match) = six_match_prediction(df)
        print("Y actual:", y_actual)
        print("Y pred:", y_pred)
        print("Date of match:", date_of_match)
        risk = int(risk_assesment(df_copy_risk, player_id))
        print("Risk:", risk)
        venue = df['venue'].iloc[0]
        print("Venue:", venue)

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
            "points": points.tolist() if isinstance(points, (np.ndarray, pd.Series)) elif float(points) else 42,
            "rank": rank.tolist() if isinstance(rank, (np.ndarray, pd.Series)) else float(rank),
            "y_actual": y_actual.tolist() if isinstance(y_actual, (np.ndarray, pd.Series)) else y_actual,
            "y_pred": y_pred.tolist() if isinstance(y_pred, (np.ndarray, pd.Series)) else y_pred,
            "mod_player_id": mod_player_id.to_dict('records') if isinstance(mod_player_id, pd.DataFrame) else [{'player_id': int(t[0]), 'y': float(t[1])} for t in mod_player_id] if isinstance(mod_player_id, np.ndarray) else mod_player_id,
            "date_of_match": [str(d) for d in date_of_match] if isinstance(date_of_match, (list, np.ndarray, pd.Series)) else str(date_of_match),
            "team1": team1.to_dict('records') if isinstance(team1, pd.DataFrame) else [{'player_id': int(t[0]), 'y': float(t[1])} for t in team1] if isinstance(team1, np.ndarray) else team1,
            "team2": team2.to_dict('records') if isinstance(team2, pd.DataFrame) else [{'player_id': int(t[0]), 'y': float(t[1])} for t in team2] if isinstance(team2, np.ndarray) else team2,
            "temp": temp,
            "risk": risk,
            "venue": venue
        }

        return response
        
    except Exception as e:
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        return {"error": str(e)}

def ai_alert(player_id, team1, team2, date):
    try:
        url = "./prod_features/data/full_dataset.csv"
        df = pd.read_csv(url)
        df = df[(df['team'] == team1) | (df['team'] == team2)]
        df = df[df['start_date'] == date]
        print("Initial DataFrame structure:")
        print(df.head())

        player_row = df[df['player_id'] == player_id]
        if player_row.empty:
            return {"error": "Player ID not found in the dataset."}

        player_name = player_row['player'].iloc[0]
        print(player_name)

        set_file_path = "./prod_features/data/full_dataset.csv"
        set_df = pd.read_csv(set_file_path)

        set_row = set_df[set_df['player_id'] == player_id]
        if set_row.empty:
            return {"error": "Player ID not found in set.csv."}

        key_cricinfo = int(set_row['key_cricinfo'].iloc[0])
        print(f"Using key_cricinfo: {key_cricinfo}")

        news_url = CricInfoScraper.fetch_url(key_cricinfo)
        if news_url:
            print(f"News URL: {news_url}")

            parsed_html = CricInfoScraper.fetch_response(news_url)
            if parsed_html:
                print("Fetching Latest News...")
                headlines = NewsScraper.fetch_content(parsed_html, max_headlines=5)

                if headlines:
                    print("Processing Headlines...")
                    result = process_multiple_headlines(headlines, player_name)

                    response = {
                        "final_rating": result["final_rating"],
                        "insights": result["insights"]
                    }
                    return response
                else:
                    return {"error": "No headlines found."}
            else:
                return {"error": "Failed to fetch parsed HTML content."}
        else:
            return {"error": "Failed to fetch news URL."}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}
