# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 00:31:53 2024

@author: anmol
"""

def scores(df):
    
    df['wickets_taken_players']=df['wickets_taken_players'].fillna("none")
    # Iterate through rows
    for index, row in df.iterrows():
        # Initialize score if not already present
        if 'score' not in df.columns:
            df['score'] = 0
    
        match_type = df['match_type'][index]
        score = df['score'][index]
    
        # Scoring criteria based on match type
        if match_type in ['IT20', 'T20']:
            # Batting scores
            score += row['runs_scored'] + row['fours'] + 2 * row['sixes']
            if row['runs_scored'] >= 100:
                score += 16
            elif row['runs_scored'] >= 50:
                score += 8
            elif row['runs_scored'] >= 30:
                score += 4
            elif row['runs_scored'] == 0:
                score -= 2
    
            # Strike rate adjustments
            if row['balls_faced'] >= 10:
                strike_rate = row['batting_strike_rate']
                if strike_rate >= 170:
                    score += 6
                elif strike_rate >= 150.01:
                    score += 4
                elif strike_rate >= 130:
                    score += 2
                elif strike_rate < 50:
                    score -= 6
                elif strike_rate < 60:
                    score -= 4
                elif strike_rate < 70:
                    score -= 2
    
            # Bowling scores
            score += row['wickets'] * 25 + row['maidens'] * 12
            if row['wickets'] == 5:
                score += 16
            elif row['wickets'] == 4:
                score += 8
            elif row['wickets'] == 3:
                score += 4
    
            # Wickets types (only iterate if there are valid entries)
            if row['wickets_taken_players'] != 0:
                score += sum(8 for mode in row['wickets_taken_players'] if mode in ['bowled', 'lbw'])
    
            # Economy rate adjustments
            if row['balls_bowled'] >= 12:
                economy = row['bowling_economy']
                if economy <= 5:
                    score += 6
                elif economy < 6:
                    score += 4
                elif economy < 7:
                    score += 2
                elif economy >= 12:
                    score -= 6
                elif economy > 11:
                    score -= 4
                elif economy > 10:
                    score -= 2
    
        elif match_type in ['ODI', 'ODM']:
            # Batting scores
            score += row['runs_scored'] + row['fours'] + 2 * row['sixes']
            if row['runs_scored'] >= 100:
                score += 8
            elif row['runs_scored'] >= 50:
                score += 4
            elif row['runs_scored'] == 0:
                score -= 3
    
            # Strike rate adjustments
            if row['balls_faced'] >= 20:
                strike_rate = row['batting_strike_rate']
                if strike_rate >= 140:
                    score += 6
                elif strike_rate >= 120:
                    score += 4
                elif strike_rate >= 100:
                    score += 2
                elif strike_rate < 30:
                    score -= 6
                elif strike_rate < 40:
                    score -= 4
                elif strike_rate < 50:
                    score -= 2
    
            # Bowling scores
            score += row['wickets'] * 25 + row['maidens'] * 4
            if row['wickets'] == 5:
                score += 8
            elif row['wickets'] == 4:
                score += 4
    
            # Wickets types
            if row['wickets_taken_players'] != 0:
                score += sum(8 for mode in row['wickets_taken_players'] if mode in ['bowled', 'lbw'])
    
            # Economy rate adjustments
            if row['balls_bowled'] >= 30:
                economy = row['bowling_economy']
                if economy <= 2.5:
                    score += 6
                elif economy < 3.5:
                    score += 4
                elif economy < 4.5:
                    score += 2
                elif economy >= 9:
                    score -= 6
                elif economy > 8:
                    score -= 4
                elif economy > 7:
                    score -= 2
    
        else:  # Other match types
            # Batting scores
            score += row['runs_scored'] + row['fours'] + 2 * row['sixes']
            if row['runs_scored'] >= 100:
                score += 8
            elif row['runs_scored'] >= 50:
                score += 4
            elif row['runs_scored'] == 0:
                score -= 4
    
            # Bowling scores
            score += row['wickets'] * 16
            if row['wickets'] == 5:
                score += 8
            elif row['wickets'] == 4:
                score += 4
    
            # Wickets types
            if row['wickets_taken_players'] != 0:
                score += sum(8 for mode in row['wickets_taken_players'] if mode in ['bowled', 'lbw'])
    
        # Fielding scores
        score += row['catches'] * 8 + row['stumpings'] * 12 + row['runouts'] * 9
        if row['catches'] >= 3:
            score += 4
    
        # Update score in the DataFrame
        df.at[index, 'score'] = score
        
        return df
