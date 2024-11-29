# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup as soup
import re
import time
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    pipeline,
    logging,
)
from scipy.spatial.distance import cosine

# Suppress transformers logging
logging.set_verbosity_error()

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DISABLE_TQDM"] = "1"

# Initialize sentiment analysis pipeline once
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1,  # Use CPU; set to 0 or CUDA device number if using GPU
)

# Initialize tokenizer and model for embeddings once
embedding_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
embedding_model = AutoModel.from_pretrained('bert-base-uncased')

class CricInfoScraper:
    """
    Base class for scraping cricket player information and related news from ESPNcricinfo.
    """

    @staticmethod
    def fetch_url(player_id):
        """
        Constructs the URL for a cricket player on ESPNcricinfo based on their player_id,
        and fetches their news page URL.
        """
        temp_url = f"https://www.espncricinfo.com/india/content/player/{player_id}.html"
        try:
            # Attempt to fetch the temporary URL
            temp_response = requests.get(temp_url)
            temp_response.raise_for_status()  # Ensure the request was successful

            # Construct the final news URL
            url = temp_response.url + "/news"
            return url
        except RequestException as e:
            # Handle any issues with fetching the URL
            print(f"Error fetching data for {temp_url}: {e}")
            return None  # Explicitly return None to indicate failure

    @staticmethod
    def fetch_response(url):
        """
        Fetches the HTML content of a given URL and parses it using BeautifulSoup.
        """
        try:
            # Make a request to the provided URL
            response = requests.get(url)
            response.raise_for_status()  # Ensure the request was successful

            # Parse the HTML content
            parsed_content = soup(response.content, 'html.parser')
            return parsed_content
        except RequestException as e:
            # Handle any issues with the request
            print(f"Error fetching response from {url}: {e}")
            return None  # Explicitly return None to indicate failure


class NewsScraper(CricInfoScraper):
    """
    Extended class for extracting cricket news articles from the parsed HTML content.
    """

    @staticmethod
    def fetch_content(parsed_content, max_headlines=5):
        """
        Extracts and returns the titles of news articles from the parsed HTML content.
        Limits the number of headlines to `max_headlines`.
        """
        headlines = []
        try:
            # Locate the parent container for all news articles
            news_container = parsed_content.find('div', class_='ds-p-4')

            if not news_container:
                print("No news container found.")
                return headlines  # Return empty list if no news container found

            # Find all individual news articles
            news_items = news_container.find_all('div', recursive=False)  # Adjust this as needed for nested divs

            # Loop through each news item and extract details
            for news in news_items[:max_headlines]:
                # Extract title
                title_tag = news.find('h2', class_='ds-text-title-s ds-font-bold ds-text-typo')
                title = title_tag.text.strip() if title_tag else None

                if title:
                    headlines.append(title)

            return headlines
        except Exception as e:
            print(f"Error while processing content: {e}")
            return headlines
        
def analyze_sentiment(headline):
    """
    Analyzes the sentiment of the headline using a pre-trained BERT model.
    Returns a sentiment score: Positive (+10), Neutral (0), or Negative (-10).
    """
    result = sentiment_analyzer(headline)[0]
    label = result['label']
    score = result['score']

    if label == 'POSITIVE':
        sentiment_score = 10 * score
    elif label == 'NEGATIVE':
        sentiment_score = -10 * score
    else:
        sentiment_score = 0

    return sentiment_score

def compute_relevance(headline, player_name):
    """
    Computes the relevance score between the headline and the player's name.
    Returns a relevance score between 0 and 10.
    """
    # Preprocess headline and player_name
    headline_lower = headline.lower()
    player_name_lower = player_name.lower()
    name_parts = player_name_lower.split()
    
    # Initialize relevance score
    relevance_score = 0.0

    # Check for direct mentions
    if player_name_lower in headline_lower:
        # Full name mentioned
        relevance_score = 10.0
    elif any(name_part in headline_lower for name_part in name_parts):
        # First name or last name mentioned
        relevance_score = 8.0
    else:
        # Proceed to compute similarity
        # Encode the texts
        inputs_headline = embedding_tokenizer(headline, return_tensors='pt', truncation=True, max_length=512)
        inputs_player = embedding_tokenizer(player_name, return_tensors='pt', truncation=True, max_length=512)

        # Get the embeddings
        with torch.no_grad():
            outputs_headline = embedding_model(**inputs_headline)
            outputs_player = embedding_model(**inputs_player)

        # Take the mean of the token embeddings
        headline_embedding = outputs_headline.last_hidden_state.mean(dim=1).numpy()[0]
        player_embedding = outputs_player.last_hidden_state.mean(dim=1).numpy()[0]

        # Compute cosine similarity
        similarity = 1 - cosine(headline_embedding, player_embedding)

        # Ensure similarity is between 0 and 1
        similarity = max(0, min(1, similarity))

        # Scale similarity to 0-5 relevance score
        relevance_score = similarity * 5  # Lower maximum relevance

    return relevance_score


def normalize_score_to_5_scale(composite_score, min_score=-10, max_score=20):
    normalized_score = 1 + 4 * (composite_score - min_score) / (max_score - min_score)
    return round(max(1, min(5, normalized_score)), 1)

def process_multiple_headlines(headlines, player_name):
    total_weighted_sentiment = 0
    total_relevance = 0
    insights = []

    for headline in headlines:
        try:
            sentiment_score = analyze_sentiment(headline)
            relevance_score = compute_relevance(headline, player_name)

            # Aggregate weighted scores
            total_weighted_sentiment += sentiment_score * relevance_score
            total_relevance += relevance_score

            # Collect insights
            insights.append(f"Headline: {headline}\nSentiment: {sentiment_score:.2f}, Relevance: {relevance_score:.2f}\n")

        except Exception as e:
            print(f"Error processing headline: {headline}\n{e}")
            continue  # Skip to the next headline

        # Pause to prevent overloading resources
        time.sleep(0.1)

    # Calculate the final composite score
    composite_score = total_weighted_sentiment / total_relevance if total_relevance > 0 else 0
    normalized_rating = normalize_score_to_5_scale(composite_score)

    return {
        "final_rating": normalized_rating,
        "composite_score": composite_score,
        "insights": insights
    }

if __name__ == "__main__":
    # Specify the player ID and name (example: Virat Kohli)
    player_id = "253802"
    player_name = "Virat Kohli"

    # Step 1: Fetch URL
    news_url = CricInfoScraper.fetch_url(player_id)
    if news_url:
        print(f"News URL: {news_url}")

        # Step 2: Fetch Parsed HTML Content
        parsed_html = CricInfoScraper.fetch_response(news_url)
        if parsed_html:
            # Step 3: Extract News Content
            print("Fetching Latest News...")
            headlines = NewsScraper.fetch_content(parsed_html, max_headlines=5)

            if headlines:
                # Step 4: Process Headlines with BERT
                print("Processing Headlines...")
                result = process_multiple_headlines(headlines, player_name)

                # Output Results
                print(f"Final Alert Score (1-5): {result['final_rating']}\n")
                print("Insights:")
                for insight in result["insights"]:
                    print(insight)
            else:
                print("No headlines found.")
        else:
            print("Failed to fetch parsed HTML content.")
    else:
        print("Failed to fetch news URL.")
        
