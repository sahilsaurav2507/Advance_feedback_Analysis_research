import pandas as pd
import numpy as np
import json
import spacy
import nltk
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
from transformers import pipeline

nltk.download('vader_lexicon')
nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()
summarizer = pipeline("summarization")

# Load dataset
df = pd.read_csv("survey_responses.csv")

def extract_keywords(text, top_n=5):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray().flatten()
    keywords = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)[:top_n]
    return [word for word, _ in keywords]

def extract_parameters(text):
    doc = nlp(text)
    entities = {ent.label_: ent.text for ent in doc.ents}
    return entities

df["keywords"] = df["response_text"].apply(extract_keywords)
df["parameters"] = df["response_text"].apply(extract_parameters)

def get_sentiment(text):
    sentiment_scores = sia.polarity_scores(text)
    return "Positive" if sentiment_scores["compound"] > 0.05 else ("Negative" if sentiment_scores["compound"] < -0.05 else "Neutral")

df["sentiment"] = df["response_text"].apply(get_sentiment)

def compute_relevance(text, keyword_list):
    if not keyword_list:
        return 0
    text_vector = TfidfVectorizer().fit_transform([text])
    keyword_vector = TfidfVectorizer().fit_transform([" ".join(keyword_list)])
    return cosine_similarity(text_vector, keyword_vector)[0][0]

df["relevance_score"] = df.apply(lambda row: compute_relevance(row["response_text"], row["keywords"]), axis=1)

df["summary"] = df["response_text"].apply(lambda x: summarizer(x, max_length=50, min_length=10, do_sample=False)[0]["summary_text"])


fig_sentiment = px.pie(df, names="sentiment", title="Sentiment Distribution in Survey Responses")
fig_sentiment.show()

plt.figure(figsize=(10,5))
plt.hist(df["relevance_score"], bins=20, color="skyblue", edgecolor="black")
plt.title("Relevance Score Distribution")
plt.xlabel("Relevance Score")
plt.ylabel("Frequency")
plt.show()

wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(sum(df["keywords"].tolist(), [])))
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

df[["response_text", "keywords", "parameters", "sentiment", "relevance_score", "summary"]].to_json("survey_analysis_results.json", orient="records", indent=4)

