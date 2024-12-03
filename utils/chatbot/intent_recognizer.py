import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class IntentRecognizer:
    def __init__(self):
        # Define the intents and their descriptions
        self.intents = {
            'batting_strike_rate': 'Retrieve the batting strike rate (runs per 100 balls) of a player.',
            'bowling_economy': 'Get the bowling economy rate (runs conceded per over) of a player.',
            'pitch_score': 'Provide the pitch score for the player at a venue.',
            'floor': 'Show the floor value of the player.',
            'ceil': 'Show the ceiling value of the player.',
            'batting_first_fantasy_points': 'Get predicted fantasy points when batting first.',
            'chasing_fantasy_points': 'Obtain predicted fantasy points when the player is batting second that is in chasing.',
            'relative_points': 'Show the relative fantasy points of the player compared to others.',
            'matchup_rank': 'Provide the matchup rank of the player.',
            'six_match_predictions': 'Get for the player’s performance in the last six matches.',
            'risk': 'Get the risk factor of the player.',
            'ai_alert': 'Show AI alerts for the player.'
        }
        
        self.intent_synonyms = {
            'strike rate': 'batting_strike_rate',
            'batting strike rate': 'batting_strike_rate',
            'batting s/r': 'batting_strike_rate',
            'economy rate': 'bowling_economy',
            'bowling economy': 'bowling_economy',
            'eco rate': 'bowling_economy',
            'pitch score': 'pitch_score',
            'floor value': 'floor',
            'ceiling value': 'ceil',
            'fantasy points batting first': 'batting_first_fantasy_points',
            'fantasy points chasing': 'chasing_fantasy_points',
            'compare fantasy points': 'relative_points',
            'relative fantasy points': 'relative_points',
            'player rank': 'matchup_rank',
            'matchup rank': 'matchup_rank',
            'last six matches': 'six_match_predictions',
            'six matches performance': 'six_match_predictions',
            'risk factor': 'risk',
            'risky comparision': 'risk',
            'player risk': 'risk',
            'ai alert': 'ai_alert',
            'ai alerts': 'ai_alert',
        }

        # Initialize the tokenizer and model
        # self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Precompute embeddings
        self.intent_embeddings = {
            intent: self.get_embedding(description)
            for intent, description in self.intents.items()
        }
            
            
    def get_embedding(self, text):
        """Generate an embedding for a given text."""
        embedding = self.model.encode(text)
        return embedding.reshape(1, -1)  # Reshape to 2D array

    def normalize_query(self, user_query):
        """Normalize the user query to a predefined intent using synonyms."""
        normalized_query = user_query.lower()
        return self.intent_synonyms.get(normalized_query, None)

    def recognize_intent(self, user_query):
        """
        Recognize the intent from the user query using either normalization or embeddings.
        """
        # First attempt: Normalize query using synonyms
        normalized_intent = self.normalize_query(user_query)
        if normalized_intent:
            return normalized_intent

        # Second attempt: Use embedding similarity
        query_embedding = self.get_embedding(user_query)
        similarities = {
            intent: cosine_similarity(query_embedding, intent_embedding)[0][0]
            for intent, intent_embedding in self.intent_embeddings.items()
        }
        recognized_intent = max(similarities, key=similarities.get)
        max_similarity = similarities[recognized_intent]

        # Apply a similarity threshold
        threshold = 0.4  # Adjust as needed
        if max_similarity >= threshold:
            return recognized_intent
        else:
            return None