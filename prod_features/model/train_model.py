# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:14:12 2024

@author: anmol
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import os
import joblib

def train_model(df):

    # ------------------ Data Preparation ------------------
    
    # Assume 'df' is your DataFrame loaded with data
    # For example:
    # df = pd.read_csv('your_data.csv')
    
    # Rename columns for consistency
    df = df[['match_type','player_id', 'start_date', 'score', 'team', 'opponent', 'venue',
              'gender']].rename(
        columns={'start_date': 'ds', 'score': 'y'}
    )
    
    # Convert 'ds' to datetime
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    
    # List of categorical columns to encode
    categorical_cols = ['match_type', 'player_id', 'team', 'opponent', 'venue', 'gender']

    # Dictionary to store the encoders for each categorical column
    encoders = {}

    # Fit and store the encoders
    for col in categorical_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])  # Transform the training data
        encoders[col] = encoder  # Store the encoder for future use

    # Specify the directory and filename to save the encoders
    directory = os.path.join(os.path.dirname(__file__), 'model_artifacts')
    os.makedirs(directory, exist_ok=True)
    encoders_path = os.path.join(directory, "label_encoders.joblib")

    # Save the encoders using joblib
    joblib.dump(encoders, encoders_path)
    
    # Extract date-related features
    df['year'] = df['ds'].dt.year
    df['month'] = df['ds'].dt.month
    df['day'] = df['ds'].dt.day
    df['day_of_week'] = df['ds'].dt.dayofweek
    
    # Optionally drop the original date column
    df.drop(columns=['ds'], inplace=True)
    
    # **Verify data types**
    print("Data types of features:")
    print(df.dtypes)
    
    # Separate features (X) and target (y)
    X = df.drop(columns=['y'])
    y = df['y']
    
    # ------------------ Model Training ------------------
    # Define and train the XGBoost regressor
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        objective='reg:squarederror',
        random_state=42
    )
    model.fit(X, y)

    # Specify the directory and filename
    model_path = os.path.join(directory, "xgboost_scores.joblib")
    
    # Save the model
    joblib.dump(model, model_path)
