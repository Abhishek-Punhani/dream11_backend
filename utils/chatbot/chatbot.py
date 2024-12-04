import os
import pandas as pd
import json
from utils.chatbot.intent_recognizer import IntentRecognizer
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# HuggingFace API Key
sec_key = os.getenv("HUGGINGFACE_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = sec_key

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Initialize the LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    temperature=0.8,
    model_kwargs={"max_length": 50},
    huggingfacehub_api_token=sec_key
)

explain_metric_prompt = PromptTemplate(
    input_variables=["metric_name", "metric_meaning"],
    template="""
You are a helpful assistant.

Explain what {metric_name} means.

Answer:
{metric_meaning}
"""
)

compare_metric_prompt = PromptTemplate(
    input_variables=["player1_name", "player2_name", "metric_name", "metric_meaning", "metric_value1", "metric_value2"],
    template="""
You are a cricket analyst assistant.

Compare the {metric_name} of {player1_name} and {player2_name}.

{metric_name}: {metric_meaning}

{player1_name} has a {metric_name} of {metric_value1}.
{player2_name} has a {metric_name} of {metric_value2}.

Explain why one might be better than the other with respect to this metric in less than 50 words.

Answer:
"""
)

general_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a chatbot specialized in answering cricket-related questions.

Question: {question}
Answer:
"""
)

data_prompt = PromptTemplate(
    input_variables=["question", "data"],
    template="""
You are a chatbot specialized in answering cricket-related questions.

Question: {question}

Answer the query based on the data {data}
"""
)

compare_metric_chain = compare_metric_prompt | llm
explain_metric_chain = explain_metric_prompt | llm
general_chain = general_prompt | llm
data_chain = data_prompt | llm

# Metric meanings dictionary
metric_meanings = {
    'batting_strike_rate': 'Batting strike rate is the number of runs a batsman scores per 100 balls faced.',
    'bowling_economy': 'Bowling economy rate is the average number of runs conceded per over by a bowler.',
    'pitch_score': 'Pitch score represents how favorable the pitch is for the player.',
    'floor': 'Floor value indicates the minimum expected performance of the player.',
    'ceil': 'Ceiling value indicates the maximum potential performance of the player.',
    'batting_first_fantasy_points': 'Predicted fantasy points when the player is batting first.',
    'chasing_first_fantasy_points': 'Predicted fantasy points when the player is chasing.',
    'relative_points': 'Relative points compare the player’s performance to others.',
    'matchup_rank': 'Matchup rank indicates how well the player performs against the current opponent.',
    'six_match_predictions': 'Predictions for the player’s performance in the last six matches.',
    'risk':'Risk tells you how consistent a player is. Low risk means reliable, while high risk could mean unpredictable but high potential!',
    'ai_alert': 'AI-generated alerts regarding the player.'
}

class Chatbot:
    def __init__(self, match_no=None):
        self.match_no = match_no
        if match_no:
            file_path = f"./data/file_{match_no}_modified.csv"
            self.player_data = pd.read_csv(file_path, dtype={'player_id': str})
            self.player_data['ai_alerts'] = self.player_data['ai_alerts'].apply(lambda x: json.loads(x.replace('""', '"')))
        else:
            self.player_data = None
        self.intent_recognizer = IntentRecognizer()
        self.player1_id = None
        self.player1_name = None
        self.player2_id = None
        self.player2_name = None

    def set_players(self, player1_id, player2_id):
        self.player1_id = player1_id
        self.player2_id = player2_id
        
        player1_row = self.player_data[self.player_data['player_id'] == self.player1_id]
        player2_row = self.player_data[self.player_data['player_id'] == self.player2_id]
        
        if player1_row.empty:
            print(f"Player ID {self.player1_id} not found in player_data.")
            return False
        self.player1_name = player1_row['player'].values[0]

        if player2_row.empty:
            print(f"Player ID {self.player2_id} not found in player_data.")
            return False
        self.player2_name = player2_row['player'].values[0]

        return True

    def get_metric_values(self, metric_name):
        player1_row = self.player_data[self.player_data['player_id'] == self.player1_id]
        player2_row = self.player_data[self.player_data['player_id'] == self.player2_id]

        # Use metric_name (with underscores) to access DataFrame columns
        if metric_name in player1_row.columns and metric_name in player2_row.columns:
            metric_value1 = player1_row[metric_name].values[0]
            metric_value2 = player2_row[metric_name].values[0]
            return metric_value1, metric_value2
        else:
            return None, None

    def process_user_query(self, user_query):
        recognized_intent = self.intent_recognizer.recognize_intent(user_query)
        if recognized_intent == 'exit':
            return 'exit', None

        if recognized_intent is None:
            return None, "I'm sorry, I didn't understand that. I'm a cricket chatbot. Please ask a relevant cricket-related question."

        # Check if the user is asking for an explanation
        explanation_keywords = ['meaning', 'mean', 'explain', 'definition', 'what is']
        if any(keyword in user_query.lower() for keyword in explanation_keywords):
            action = 'explain'
        else:
            action = 'compare'

        # Use intent for data access and metric_name_display for user-friendly output
        metric_name_display = recognized_intent.replace('_', ' ')

        # Get metric meaning
        metric_meaning = metric_meanings.get(recognized_intent, 'No explanation available.')

        if action == 'compare':
            if recognized_intent == 'ai_alert':
                # Handle AI alert comparison
                player1_alerts = self.player_data[self.player_data['player_id'] == self.player1_id]['ai_alerts'].values[0]
                player2_alerts = self.player_data[self.player_data['player_id'] == self.player2_id]['ai_alerts'].values[0]
                response = f"Player 1 AI Alerts: {player1_alerts}\nPlayer 2 AI Alerts: {player2_alerts}\n\nI'm a cricket chatbot. Please ask relevant questions."
                return recognized_intent, response.strip()

            metric_value1, metric_value2 = self.get_metric_values(recognized_intent)
            if metric_value1 is not None and metric_value2 is not None:
                # Prepare inputs for the chain
                chain_inputs = {
                    "player1_name": self.player1_name,
                    "player2_name": self.player2_name,
                    "metric_name": metric_name_display,
                    "metric_meaning": metric_meaning,
                    "metric_value1": metric_value1,
                    "metric_value2": metric_value2
                }
                response = compare_metric_chain.invoke(chain_inputs)
                summary = f"{self.player1_name} has a {metric_name_display} of {metric_value1}, while {self.player2_name} has a {metric_name_display} of {metric_value2}. Based on this metric, {'player1' if metric_value1 > metric_value2 else 'player2'} is better."
                return recognized_intent, f"{response.strip()}\n\nSummary: {summary}"
            else:
                return recognized_intent, f"Data for {metric_name_display} is not available for comparison."
        elif action == 'explain':
            chain_inputs = {
                "metric_name": metric_name_display,
                "metric_meaning": metric_meaning
            }
            response = explain_metric_chain.invoke(chain_inputs)
            return recognized_intent, response.strip()
        else:
            return None, "Could you please be more specific or elaborate more on your cricket player comparison related query?"

    def handle_query(self, player1_id=None, player2_id=None, user_query=None):
        if player1_id and player2_id and self.match_no:
            if not self.set_players(player1_id, player2_id):
                return {"error": "Failed to set players. Please check the player IDs."}

            intent, response = self.process_user_query(user_query)
            if intent == 'exit':
                return {"response": "Goodbye!"}
            if response:
                return {"response": response}
            
            # If intent is None or response is not generated
            player1_data = self.player_data[self.player_data['player_id'] == self.player1_id].to_dict(orient='records')[0]
            player2_data = self.player_data[self.player_data['player_id'] == self.player2_id].to_dict(orient='records')[0]
            data = {"player1_data": player1_data, "player2_data": player2_data}
            response = data_chain.invoke({"question": user_query, "data": json.dumps(data)})
            return {"response": response.strip()}
        else:
            # Handle general queries
            if user_query.lower() in ["hi", "hello"]:
                return {"response": "Hi, how can I help you?"}
            response = general_chain.invoke({"question": user_query})
            return {"response": f"{response.strip()}\n\nI'm a cricket chatbot. Please ask relevant questions."}