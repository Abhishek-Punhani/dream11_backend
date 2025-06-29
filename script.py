import requests
import pandas as pd
import numpy as np
import json

def update_csv(file_path, output_path, api_url):
    try:
        # Step 1: Read the existing CSV
        df = pd.read_csv(file_path)
        print("Initial DataFrame structure:")
        print(df.head())
        
        # Delete the 'values' column if it exists
        if 'values' in df.columns:
            df.drop(columns=['values'], inplace=True)
        
        # Add a new column for storing JSON response and initialize with NaN
        df['values'] = np.nan

        # Step 2: Loop through each player in the DataFrame and make a POST request
        for index, row in df.iterrows():
            player_id = row['player_id']  # Ensure your CSV has a 'player_id' column
            post_data = {"player_id": player_id, "match_no": 3}
            
            try:
                print(f"\nProcessing player {player_id}...")
                
                # Send POST request
                response = requests.post(api_url, json=post_data)
                response.raise_for_status()
                player_data = response.json()  # Parse the response
                print(f"JSON response for player {player_id}: {player_data}")
                
                # Add the JSON response to the 'values' column
                df.at[index, 'values'] = json.dumps(player_data)  # Store as JSON string
                
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
        '/home/manav/dev_ws/src/dream11_backend/prod_features/data/file_3.csv',
        '/home/manav/dev_ws/src/dream11_backend/data/file_3.csv',
        'http://127.0.0.1:5000/api/user_data', 
    )