import requests
import pandas as pd
import numpy as np

def update_csv(file_path, output_path, api_url):
    try:
        # Step 1: Read the existing CSV
        df = pd.read_csv(file_path)
        # Add a new column for storing JSON response if it doesn't exist
        if 'ai_alerts' not in df.columns:
            df['ai_alerts'] = np.nan

        # Step 2: Loop through each player in the DataFrame and make a POST request
        for index, row in df.iterrows():
            player_id = row['player_id']  # Ensure your CSV has a 'player_id' column
            post_data = {"player_id": player_id,"match_no":1}
            
            try:
                print(f"\nProcessing player {player_id}...")
                
                # Send POST request
                response = requests.post(api_url, json=post_data)
                response.raise_for_status()
                player_data = response.json()  # Parse the response
                
                # Add the JSON response to the 'values' column
                df.at[index, 'ai_alerts'] = str(player_data)  # Store as string to avoid DataFrame issues
                
                print(f"Successfully processed player {player_id}")
                
            except requests.RequestException as e:
                print(f"Error fetching data for player {player_id}: {str(e)}")
                continue
            except Exception as e:
                print(f"Error processing player {player_id}: {str(e)}")
                continue
        
        # Step 3: Save updated DataFrame
        df.to_csv(output_path, index=False)
        print(f"\nUpdated CSV saved to {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

