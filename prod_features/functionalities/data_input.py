# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 20:00:21 2024

@author: anmol
"""

import pandas as pd
import os

# Function to process and save the data
def process_and_save_csv(input_csv_path):
    """
    Reads a CSV file, processes it, and saves it to a specific directory.

    Parameters:
    - input_csv_path (str): Path to the input CSV file.
    - output_dir (str): Directory to save the processed file.
    - output_file_name (str): Name of the processed file to be saved.

    Returns:
    - str: Full path to the saved file.
    """
    try:
        # Step 1: Read the CSV file
        print(f"Reading CSV file from: {input_csv_path}")
        df = pd.read_csv(input_csv_path)
        
        
        df2=pd.read_csv(r'C:\Users\anmol\cricket_analysis\data\storage_dataset.csv')


        # Merge df and df2 using all columns in df1 as the primary key
        merged_df = pd.merge(df, df2, on=['player_id','team','opponent','start_date'], how='left')
        
        

        # Step 2: Process the data (Example: Display first 5 rows)
        print("Preview of the data:")
        print(df.head())
        
        


        # Step 3: Ensure the output directory exists
        output_dir=r'C:\Users\anmol\cricket_analysis\data'
        output_file_name=r'test_data.csv'
        os.makedirs(output_dir, exist_ok=True)

        # Step 4: Save the processed file
        output_path = os.path.join(output_dir, output_file_name)
        merged_df.to_csv(output_path, index=False)

        print(f"File saved successfully at: {output_path}")
        return output_path

    except Exception as e:
        print(f"An error occurred: {e}")
        raise
