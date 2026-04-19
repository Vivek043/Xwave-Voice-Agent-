from transformers import pipeline

# Load the sentiment analysis pipeline with the pre-trained model
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Analyze sentiment
result = sentiment_pipeline("I love this product!")
print(result)