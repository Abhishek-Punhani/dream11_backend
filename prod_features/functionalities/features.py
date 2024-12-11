# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 22:49:50 2024

@author: anmol
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'prod_features')))

import pandas as pd
import joblib
from data_preprocessing.score_calculation import scores
from model.predict_model import predict_model

def batting_strike_rate(df):
    strike_rate=predict_model(df,'batting_strike_rate','')
    return strike_rate

def bowling_economy(df):
    economy=predict_model(df,'bowling_economy_rate','')
    return economy

def pitch_score(player_id,venue):
    df=pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'full_dataset.csv'))
    df=df[df['venue']==venue]
    ppi=df[df['player_id']==player_id]['score'].mean()
    ppimin=df[df['player_id']==player_id]['score'].min()
    ppimax=df[df['player_id']==player_id]['score'].max()
    return 1+((ppi-ppimin)/(ppimax-ppimin))*99

def floor(player_id):
    df=pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'full_dataset.csv'))
    player_id_dataframe={player_id:player_df for player_id,player_df in df.groupby('player_id')}
    avg_score=player_id_dataframe[player_id]['score'].mean()
    std_score=player_id_dataframe[player_id]['score'].std()
    return avg_score-std_score

def ceil(player_id):
    df=pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'full_dataset.csv'))
    player_id_dataframe={player_id:player_df for player_id,player_df in df.groupby('player_id')}
    avg_score=player_id_dataframe[player_id]['score'].mean()
    std_score=player_id_dataframe[player_id]['score'].std()
    return avg_score+std_score

def batting_first_fantasy_points(player_id,df1):
    df=pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'full_dataset.csv'))
    df=df.sort_values(by='start_date')
    df=df[df['player_id']==player_id]
    df=df.tail(3)
    original_score=df['score']
    df = df[['match_type','player_id', 'start_date', 'team', 'opponent', 'venue',
              'gender']].rename(
        columns={'start_date': 'ds'}
    )
    df1 = df1[['match_type','player_id', 'start_date', 'team', 'opponent', 'venue',
              'gender']].rename(
        columns={'start_date': 'ds'}
    )
    df=pd.concat([df,df1],axis=0,ignore_index=True)
    predicted_score=predict_model(df,'batting_first_fantasy_points','')
    return (original_score,predicted_score)

def chasing_first_fantasy_points(player_id,df1):
    df=pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'full_dataset.csv'))
    df=df.sort_values(by='start_date')
    df=df[df['player_id']==player_id]
    df=df.tail(3)
    original_score=df['score']
    df = df[['match_type','player_id', 'start_date', 'team', 'opponent', 'venue',
              'gender']].rename(
        columns={'start_date': 'ds'}
    )
    df1 = df1[['match_type','player_id', 'start_date', 'team', 'opponent', 'venue',
              'gender']].rename(
        columns={'start_date': 'ds'}
    )
    df=pd.concat([df,df1],axis=0,ignore_index=True)
    predicted_score=predict_model(df,'chasing_first_fantasy_points','')
    return (original_score,predicted_score)
    

def relative_points(df,player_id):
    # Load the saved LabelEncoder during inference
    encoder_path = os.path.join(os.path.dirname(__file__), '..', 'model_artifacts', 'label_encoders.joblib')
    loaded_encoder = joblib.load(encoder_path)  
    encoded_player_id = loaded_encoder['player_id'].transform([player_id])[0]
    relative_points=predict_model(df,'relative_points',encoded_player_id)
    return relative_points

def matchup_rank(df,player_id):
    df1=pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'full_dataset.csv'))
    dicti={}
    for index,row in df.iterrows():
        df1=df1[df1['player_id']==row['player_id']]
        df1=df1[df1['opponent']==row['opponent']]
        dicti[row['player_id']]=df1['score'].mean()
        
    sorted_dict=sorted(dicti.items(),key=lambda items:items[1])
    # Step 2: Find the index of a specific key (e.g., 'b')
    key_to_find = player_id

    # Iterate over the sorted items to find the index
    for idx, (key, value) in enumerate(sorted_dict):
      if key == key_to_find:
        return idx+1

def six_match_prediction(df):
    player_id=df['player_id'].iloc[0]
    target_date=df['start_date'].iloc[0]
    df1=pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'full_dataset.csv'))
    df1=df1.sort_values(by='start_date')
    # Convert the 'date' column to datetime
    df1['start_date'] = pd.to_datetime(df1['start_date'])

    # Given target date to filter against
    target_date = pd.to_datetime(target_date)

    # Step 1: Filter the DataFrame where date is less than the given target date
    filtered_df = df1[df1['start_date'] < target_date]
    filtered_df=filtered_df[filtered_df['player_id']==player_id]

    # Step 2: Get the first 6 rows
    result_df = filtered_df.tail(6)
    y_pred=predict_model(result_df,'predictions', '')
    y_actual=result_df['score']
    return (y_actual,y_pred,target_date)

def transaction_rating(team1,team2,temp,player_id):
    # Load the saved LabelEncoder during inference
    encoder_path = os.path.join(os.path.dirname(__file__), '..', 'model_artifacts', 'label_encoders.joblib')
    loaded_encoder = joblib.load(encoder_path) 
    
    encoded_player_id = loaded_encoder['player_id'].transform([player_id])[0]
    df_t=team2[team2['player_id']==encoded_player_id]
    temp+=df_t['y'].iloc[0]
    team1=pd.concat([team1,df_t],axis=0,ignore_index=True)
    team1=team1.sort_values(by='y',ascending=False)
    row=team1[11:12]
    team1=team1[:11]
    team2=pd.concat([team2,row],axis=0,ignore_index=True)
    team2.sort_values(by='y',ascending=False)
    temp-=row['y']
    return (team1,team2,temp)
    

def team_spider_chart(df,team1,dream_team_points):
    fes=0
    # Load the saved LabelEncoder during inference
    encoder_path = os.path.join(os.path.dirname(__file__), '..', 'model_artifacts', 'label_encoders.joblib')
    loaded_encoder = joblib.load(encoder_path) 
    # Transform the categorical data using the corresponding encoder
    df['player_id'] = loaded_encoder['player_id'].transform(df['player_id'])
    
    df = df[df['player_id'].isin(team1['player_id'])]

    fes+=df['runouts'].sum()+df['catches'].sum()+df['stumpings'].sum()
    
    pcb=1-(team1['y'].std())/(team1['y'].mean())
    
    doi=0
    
    df.drop(columns=['runs_scored','fours','sixes','wickets'])
    
    df.rename(columns={'runs_death':'runs_scored','fours_death':'fours','sixes_death':'sixes','wickets_death':'wickets'})
    
    df=scores(df)
        
    doi=df['score'].sum()/dream_team_points
    
    return (fes,doi,pcb)
                      



def risk_assesment(df, player_id):
    # Create a new column 'risk' in the DataFrame
    df=scores(df)
    df=df.fillna(0)
    df = df[df['player_id']==player_id]
    risk = (df['score'].std())
    if pd.isna(risk):
        return -1
    return risk if risk < 100 else 100