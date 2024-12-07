from flask import jsonify, request
import pandas as pd
import polars as pl
import numpy as np
from prod_features.model.predict_model import predict_model
from prod_features.functionalities.features import (
    batting_strike_rate, bowling_economy, pitch_score, floor, ceil,
    batting_first_fantasy_points, chasing_first_fantasy_points,
    relative_points, matchup_rank, six_match_prediction,
     team_spider_chart,risk_assesment
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
        url2=f"./prod_features/data/full_dataset.csv"
        df = pd.read_csv(url)
        df = df[['player', 'player_id', 'team', 'opponent', 'start_date', 'end_date', 'venue', 'match_type', 'is_captain', 'is_player_of_match', 'runs_scored', 'balls_faced', 'outs', 'fours', 'sixes', 'runs_conceded', 'balls_bowled', 'wickets', 'catches', 'stumpings', 'runouts', 'wickets_taken_players', 'dismissed_by', 'wicket_type', 'runs_powerplay', 'fours_powerplay', 'sixes_powerplay', 'wickets_powerplay', 'runs_middle', 'fours_middle', 'sixes_middle', 'wickets_middle', 'runs_death', 'fours_death', 'sixes_death', 'wickets_death', 'win', 'win_by_run', 'win_by_wickets', 'maidens', 'gender', 'batting_average', 'batting_strike_rate', 'bowling_average', 'bowling_strike_rate', 'bowling_economy']]
       
        df_copy = df.copy()
        df_copy_risk=pd.read_csv(url2)
        df = df[df['player_id'] == player_id]
        
        # Get all values
        (y_pred_sorted, dream_team_points,mod_player_id) = predict_model(df_copy, 'match_predictions', '')
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
       
        fes = int(fes) if isinstance(fes, np.integer) else fes
        doi = float(doi) if isinstance(doi, np.floating) else doi
        pcb = float(pcb) if isinstance(pcb, np.floating) else pcb
        risk=int(risk_assesment(df_copy_risk,player_id))
        venue=df['venue'].iloc[0]
        

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
            "mod_player_id": mod_player_id.to_dict('records') if isinstance(mod_player_id, pd.DataFrame) else [{'player_id': int(t[0]), 'y': float(t[1])} for t in mod_player_id] if isinstance(mod_player_id, np.ndarray) else mod_player_id,
            "date_of_match": [str(d) for d in date_of_match] if isinstance(date_of_match, (list, np.ndarray, pd.Series)) else str(date_of_match),
            "team1": team1.to_dict('records') if isinstance(team1, pd.DataFrame) else [{'player_id': int(t[0]), 'y': float(t[1])} for t in team1] if isinstance(team1, np.ndarray) else team1,
            "team2": team2.to_dict('records') if isinstance(team2, pd.DataFrame) else [{'player_id': int(t[0]), 'y': float(t[1])} for t in team2] if isinstance(team2, np.ndarray) else team2,
            "temp": temp,
            "pfpp": pfpp,
            "fes": fes,
            "doi": doi,
            "pcb": pcb,
            "risk":risk,
            "venue":venue
          
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
    

def generate_fantasy_points_data(filtered_df, player_id):
    # Ensure the required columns are present
    required_columns = ['start_date', 'floor', 'ceiling', 'batting_first_fp', 'chasing_fp']
    for col in required_columns:
        if col not in filtered_df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame")

    # Convert start_date to string to ensure JSON serializability
    filtered_df = filtered_df.with_columns(
        pl.col('start_date').cast(pl.Utf8)
    )

    return pd.DataFrame({
        'start_date': filtered_df['start_date'].to_list(),
        'floor': filtered_df['floor'].to_list(),
        'ceiling': filtered_df['ceiling'].to_list(),
        'batting_first_fp': filtered_df['batting_first_fp'].to_list(),
        'chasing_fp': filtered_df['chasing_fp'].to_list()
    })

def determine_batting_order(df):
    # Determine batting order based on 'win_by_run' and 'win_by_wickets'
    df = df.with_columns(
        pl.when(pl.col('win_by_run') > 0)
        .then(pl.lit('Batting First'))
        .when(pl.col('win_by_wickets') > 0)
        .then(pl.lit('Chasing'))
        .otherwise(pl.lit('Unknown'))
        .alias('batting_order')
    )
    return df

def calculate_metrics(df):
    window_size = 3

    # Calculate rolling average and standard deviation
    df = df.with_columns([
        pl.col("Dream11_Points").rolling_mean(window_size).alias("Rolling_Avg"),
        pl.col("Dream11_Points").rolling_std(window_size).alias("Rolling_Std")
    ])

    # Calculate floor and ceiling
    df = df.with_columns([
        (pl.col("Rolling_Avg") - 1.96 * pl.col("Rolling_Std")).alias("floor"),
        (pl.col("Rolling_Avg") + 1.96 * pl.col("Rolling_Std")).alias("ceiling")
    ])

    # Create conditional columns for Batting First and Chasing Fantasy Points
    df = df.with_columns([
        pl.when(pl.col("batting_order") == "Batting First")
        .then(pl.col("Dream11_Points"))
        .otherwise(pl.lit(0))
        .alias("Batting_First_Dream11_Points"),
        
        pl.when(pl.col("batting_order") == "Chasing")
        .then(pl.col("Dream11_Points"))
        .otherwise(pl.lit(0))
        .alias("Chasing_Dream11_Points")
    ])

    # Calculate rolling sums for Batting First Fantasy Points
    df = df.with_columns([
        pl.col("Batting_First_Dream11_Points").rolling_sum(window_size).alias("batting_first_fp"),
        pl.col("Chasing_Dream11_Points").rolling_sum(window_size).alias("chasing_fp")
    ])

    return df

def get_fantasy_points():
    data = request.get_json()
    player_id = data.get('player_id')

    # Define the dataset path
    dataset_path = r'/home/manav/dev_ws/src/dream11_backend/filtered_dataset.csv'  # Update this path as needed

    # Step 2: Load the dataset
    try:
        df = pl.read_csv(dataset_path)
    except FileNotFoundError:
        return jsonify({"error": f"The file '{dataset_path}' was not found."}), 404
    except Exception as e:
        return jsonify({"error": f"An error occurred while reading the CSV file: {e}"}), 500

    # Step 3: Parse 'start_date' column to datetime
    if 'start_date' in df.columns:
        try:
            df = df.with_columns(
                pl.col("start_date").str.strptime(pl.Date, "%Y-%m-%d").alias("start_date")
            )
        except Exception as e:
            return jsonify({"error": f"Error parsing 'start_date': {e}"}), 500
    else:
        return jsonify({"error": "'start_date' column not found in the dataset."}), 400

    # Step 4: Filter the DataFrame based on player_id (case-insensitive)
    filtered_df = df.filter(pl.col('player_id') == player_id)

    if filtered_df.is_empty():
        return jsonify({"error": f"No data found for the specified Player ID '{player_id}'. Please check the ID and try again."}), 404

    # Step 5: Sort by start_date
    filtered_df = filtered_df.sort('start_date')

    # Step 6: Determine batting order
    filtered_df = determine_batting_order(filtered_df)

    # Step 7: Calculate metrics
    filtered_df = calculate_metrics(filtered_df)

    # Step 8: Generate Fantasy Points metrics data
    data_points = generate_fantasy_points_data(filtered_df, player_id)

    # Step 9: Check if data_points is not empty
    if not data_points.empty:
        # Add player_id and player_name to the data points
       
        data_points['player_id'] = player_id
       
        # Convert to JSON response with arrays for the metrics
        response = {
           
            "player_id": player_id,
            "start_date": data_points['start_date'].tolist(),
            "floor": data_points['floor'].tolist(),
            "ceiling": data_points['ceiling'].tolist(),
            "batting_first_fp": data_points['batting_first_fp'].tolist(),
            "chasing_fp": data_points['chasing_fp'].tolist()
        }
        return jsonify(response), 200
    else:
        return jsonify({"error": f"No data to display for player ID '{player_id}'."}), 404
