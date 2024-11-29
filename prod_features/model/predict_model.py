# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:14:09 2024

@author: anmol
"""

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder


def predict_model(df, str, player_id):

    if str == 'batting_strike_rate':

        # Rename columns for consistency
        df = df[['match_type', 'player_id', 'start_date', 'team', 'opponent', 'venue',
                 'gender']].rename(
            columns={'start_date': 'ds'}
        )

        # Convert 'ds' to datetime
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')

        # Load the saved LabelEncoder during inference
        loaded_encoder = joblib.load(
            r'C:\Users\anmol\prod_features\model_artifacts\label_encoders.joblib')

        # Example: Use the loaded encoder to transform new labels during inference
        # List of categorical columns to encode
        categorical_cols = ['match_type', 'player_id',
                            'team', 'opponent', 'venue', 'gender']
        for col in categorical_cols:
            if col in loaded_encoder:
                # Transform the categorical data using the corresponding encoder
                df[col] = loaded_encoder[col].transform(df[col])

        # Extract date-related features
        df['year'] = df['ds'].dt.year
        df['month'] = df['ds'].dt.month
        df['day'] = df['ds'].dt.day
        df['day_of_week'] = df['ds'].dt.dayofweek

        # Optionally drop the original date column
        df.drop(columns=['ds'], inplace=True)

        # Separate features (X) and target (y)
        X = df

        # Load the saved model
        model = joblib.load(
            r'C:\Users\anmol\prod_features\model_artifacts\xgboost_batting_strike_rate.joblib')

        # ------------------ Prediction ------------------
        # Make predictions
        y_pred = model.predict(X)
        return y_pred

    elif str == 'bowling_economy_rate':
        # Rename columns for consistency
        df = df[['match_type', 'player_id', 'start_date', 'team', 'opponent', 'venue',
                 'gender']].rename(
            columns={'start_date': 'ds'}
        )

        # Convert 'ds' to datetime
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')

        # Load the saved LabelEncoder during inference
        loaded_encoder = joblib.load(
            r'C:\Users\anmol\prod_features\model_artifacts\label_encoders.joblib')

        # Example: Use the loaded encoder to transform new labels during inference
        # List of categorical columns to encode
        categorical_cols = ['match_type', 'player_id',
                            'team', 'opponent', 'venue', 'gender']
        for col in categorical_cols:
            if col in loaded_encoder:
                # Transform the categorical data using the corresponding encoder
                df[col] = loaded_encoder[col].transform(df[col])

        # Extract date-related features
        df['year'] = df['ds'].dt.year
        df['month'] = df['ds'].dt.month
        df['day'] = df['ds'].dt.day
        df['day_of_week'] = df['ds'].dt.dayofweek

        # Optionally drop the original date column
        df.drop(columns=['ds'], inplace=True)

        # Separate features (X) and target (y)
        X = df

        # Load the saved model
        model = joblib.load(
            r'C:\Users\anmol\prod_features\model_artifacts\xgboost_bowling_economy.joblib')

        # ------------------ Prediction ------------------
        # Make predictions
        y_pred = model.predict(X)
        return y_pred

    elif str == 'batting_first_fantasy_points':
        # Rename columns for consistency

        # Convert 'ds' to datetime
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')

        # Load the saved LabelEncoder during inference
        loaded_encoder = joblib.load(
            r'C:\Users\anmol\prod_features\model_artifacts\label_encoders.joblib')

        # Example: Use the loaded encoder to transform new labels during inference
        # List of categorical columns to encode
        categorical_cols = ['match_type', 'player_id',
                            'team', 'opponent', 'venue', 'gender']
        for col in categorical_cols:
            if col in loaded_encoder:
                # Transform the categorical data using the corresponding encoder
                df[col] = loaded_encoder[col].transform(df[col])

        # Extract date-related features
        df['year'] = df['ds'].dt.year
        df['month'] = df['ds'].dt.month
        df['day'] = df['ds'].dt.day
        df['day_of_week'] = df['ds'].dt.dayofweek

        # Optionally drop the original date column
        df.drop(columns=['ds'], inplace=True)

        # Separate features (X) and target (y)
        X = df

        # Load the saved model
        model = joblib.load(
            r'C:\Users\anmol\prod_features\model_artifacts\xgboost_first_inning.joblib')

        # ------------------ Prediction ------------------
        # Make predictions
        y_pred = model.predict(X)
        return y_pred

    elif str == 'chasing_first_fantasy_points':
        # Rename columns for consistency

        # Convert 'ds' to datetime
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')

        # Load the saved LabelEncoder during inference
        loaded_encoder = joblib.load(
            r'C:\Users\anmol\prod_features\model_artifacts\label_encoders.joblib')

        # Example: Use the loaded encoder to transform new labels during inference
        # List of categorical columns to encode
        categorical_cols = ['match_type', 'player_id',
                            'team', 'opponent', 'venue', 'gender']
        for col in categorical_cols:
            if col in loaded_encoder:
                # Transform the categorical data using the corresponding encoder
                df[col] = loaded_encoder[col].transform(df[col])

        # Extract date-related features
        df['year'] = df['ds'].dt.year
        df['month'] = df['ds'].dt.month
        df['day'] = df['ds'].dt.day
        df['day_of_week'] = df['ds'].dt.dayofweek

        # Optionally drop the original date column
        df.drop(columns=['ds'], inplace=True)

        # Separate features (X) and target (y)
        X = df

        # Load the saved model
        model = joblib.load(
            r'C:\Users\anmol\prod_features\model_artifacts\xgboost_second_inning.joblib')

        # ------------------ Prediction ------------------
        # Make predictions
        y_pred = model.predict(X)
        return y_pred

    elif str == 'relative_points':

        # Rename columns for consistency
        df = df[['match_type', 'player_id', 'start_date', 'team', 'opponent', 'venue',
                 'gender']].rename(
            columns={'start_date': 'ds'}
        )
        # Convert 'ds' to datetime
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')

        # Load the saved LabelEncoder during inference
        loaded_encoder = joblib.load(
            r'C:\Users\anmol\prod_features\model_artifacts\label_encoders.joblib')

        # Example: Use the loaded encoder to transform new labels during inference
        # List of categorical columns to encode
        categorical_cols = ['match_type', 'player_id',
                            'team', 'opponent', 'venue', 'gender']
        for col in categorical_cols:
            if col in loaded_encoder:
                # Transform the categorical data using the corresponding encoder
                df[col] = loaded_encoder[col].transform(df[col])

        # Extract date-related features
        df['year'] = df['ds'].dt.year
        df['month'] = df['ds'].dt.month
        df['day'] = df['ds'].dt.day
        df['day_of_week'] = df['ds'].dt.dayofweek

        # Optionally drop the original date column
        df.drop(columns=['ds'], inplace=True)

        # Separate features (X) and target (y)
        X = df

        # Load the saved model
        model = joblib.load(
            r'C:\Users\anmol\prod_features\model_artifacts\xgboost_score.joblib')

        # ------------------ Prediction ------------------
        # Make predictions
        y_pred = model.predict(X)

        y_pred = pd.concat(
            [X['player_id'], pd.Series(y_pred, name='y')], axis=1)

        y_pred_sorted = y_pred.sort_values(by='y', ascending=False)

        dream_team_points_predicted = 0

        for i in range(len(y_pred)):
            if i < 11:
                dream_team_points_predicted += y_pred_sorted.iloc[i]['y']
            else:
                break
        df = y_pred[y_pred['player_id'] == player_id]

        return df['y'].iloc[0]/dream_team_points_predicted
    elif str=='predictions':
        # Rename columns for consistency
        df = df[['match_type', 'player_id', 'start_date', 'team', 'opponent', 'venue',
                 'gender']].rename(
            columns={'start_date': 'ds'}
        )

        # Load the saved LabelEncoder during inference
        loaded_encoder = joblib.load(
            r'C:\Users\anmol\prod_features\model_artifacts\label_encoders.joblib')

        # Example: Use the loaded encoder to transform new labels during inference
        # List of categorical columns to encode
        categorical_cols = ['match_type', 'player_id',
                            'team', 'opponent', 'venue', 'gender']
        for col in categorical_cols:
            if col in loaded_encoder:
                # Transform the categorical data using the corresponding encoder
                df[col] = loaded_encoder[col].transform(df[col])

        # Extract date-related features
        df['year'] = df['ds'].dt.year
        df['month'] = df['ds'].dt.month
        df['day'] = df['ds'].dt.day
        df['day_of_week'] = df['ds'].dt.dayofweek

        # Optionally drop the original date column
        df.drop(columns=['ds'], inplace=True)

        # Separate features (X) and target (y)
        X = df

        # Load the saved model
        model = joblib.load(
            r'C:\Users\anmol\prod_features\model_artifacts\xgboost_score.joblib')

        # ------------------ Prediction ------------------
        # Make predictions
        y_pred = model.predict(X)
        return y_pred
    
    elif str=='match_predictions':
        # Rename columns for consistency
        df = df[['match_type', 'player_id', 'start_date', 'team', 'opponent', 'venue',
                 'gender']].rename(
            columns={'start_date': 'ds'}
        )
        # Convert 'ds' to datetime
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    
        # Load the saved LabelEncoder during inference
        loaded_encoder = joblib.load(
            r'C:\Users\anmol\prod_features\model_artifacts\label_encoders.joblib')
    
        # Example: Use the loaded encoder to transform new labels during inference
        # List of categorical columns to encode
        categorical_cols = ['match_type', 'player_id',
                            'team', 'opponent', 'venue', 'gender']
        for col in categorical_cols:
            if col in loaded_encoder:
                # Transform the categorical data using the corresponding encoder
                df[col] = loaded_encoder[col].transform(df[col])
    
        # Extract date-related features
        df['year'] = df['ds'].dt.year
        df['month'] = df['ds'].dt.month
        df['day'] = df['ds'].dt.day
        df['day_of_week'] = df['ds'].dt.dayofweek
    
        # Optionally drop the original date column
        df.drop(columns=['ds'], inplace=True)
    
        # Separate features (X) and target (y)
        X = df
    
        # Load the saved model
        model = joblib.load(
            r'C:\Users\anmol\prod_features\model_artifacts\xgboost_score.joblib')
    
        # ------------------ Prediction ------------------
        # Make predictions
        y_pred = model.predict(X)
    
        y_pred = pd.concat(
            [X['player_id'], pd.Series(y_pred, name='y')], axis=1)
    
        y_pred_sorted = y_pred.sort_values(by='y', ascending=False)
    
        dream_team_points_predicted = 0
    
        for i in range(len(y_pred)):
            if i < 11:
                dream_team_points_predicted += y_pred_sorted.iloc[i]['y']
            else:
                break
    
    
    return (y_pred_sorted,dream_team_points_predicted)
