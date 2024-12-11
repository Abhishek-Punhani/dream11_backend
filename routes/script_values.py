import requests
import pandas as pd
import numpy as np
import json

def update_csv(file_path, output_path, api_url):
    try:
        # Step 1: Read the existing CSV
        df = pd.read_csv(file_path)
        df = df[['player', 'player_id', 'team', 'opponent', 'start_date', 'end_date', 'venue', 'match_type', 'is_captain', 'is_player_of_match', 'runs_scored', 'balls_faced', 'outs', 'fours', 'sixes', 'runs_conceded', 'balls_bowled', 'wickets', 'catches', 'stumpings', 'runouts', 'wickets_taken_players', 'dismissed_by', 'wicket_type', 'runs_powerplay', 'fours_powerplay', 'sixes_powerplay', 'wickets_powerplay', 'runs_middle', 'fours_middle', 'sixes_middle', 'wickets_middle', 'runs_death', 'fours_death', 'sixes_death', 'wickets_death', 'win', 'win_by_run', 'win_by_wickets', 'maidens', 'gender', 'batting_average', 'batting_strike_rate', 'bowling_average', 'bowling_strike_rate', 'bowling_economy']]
        print("Initial DataFrame structure:")
        print(df.head())

        # df1=pd.read_csv('/home/manav/dev_ws/src/dream11_backend/data/New/file_1.csv')

        
        # Delete the 'values' column if it exists
        if 'values' in df.columns:
            df.drop(columns=['values'], inplace=True)

        # if 'y_predicted' in df.columns:
        #     df.drop(columns=['y_predicted'], inplace=True)
        
        # Add a new column for storing JSON response and initialize with NaN
        df['values'] = np.nan
        # df['y_predicted'] = np.nan

        # Step 2: Loop through each player in the DataFrame and make a POST request
        for index, row in df.iterrows():
            player_id = row['player_id']  # Ensure your CSV has a 'player_id' column
            post_data = {"player_id": player_id, "match_no": 6}
            
            try:
                print(f"\nProcessing player {player_id}...")
                
                # Send POST request
                response = requests.post(api_url, json=post_data)
                response.raise_for_status()
                player_data = response.json()  # Parse the response
                print(f"JSON response for player {player_id}: {player_data}")
                print(f"JSON response for player {player_id}: {player_data}")
                
                # Add the JSON response to the 'values' column
                df.at[index, 'values'] = json.dumps(player_data)  # Store as JSON string
                # df.at[index, 'y_predicted'] = df1['y_pred']
                
                print(f"Successfully processed player {player_id}")
                
            except requests.RequestException as e:
                print(f"Error fetching data for player {player_id}: {str(e)}")
                continue
            except Exception as e:
                print(f"Error processing player {player_id}: {str(e)}")
                continue
        
        # Step 3: Save updated DataFrame
        print("Final DataFrame structure:")
        print(df.head())
        df.to_csv(output_path, index=False)
        print(f"\nUpdated CSV saved to {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    update_csv(
        '/home/manav/dev_ws/src/dream11_backend/prod_features/data/file_6.csv',
        '/home/manav/dev_ws/src/dream11_backend/data/file_6_final.csv',
        'http://127.0.0.1:5000/api/user_data', 
    )