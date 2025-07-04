'''
Analysing the customer complaints the bank has received over the past year. The goal is to use NLP techniques, 
such as text classification and sentiment analysis, to efficiently gain insights into the underlying causes of 
customer grievances. By leveraging these methods, we aim to better understand and address customer grievances, 
ultimately improving our grievance redressal process.

'''

# 1. Load and Explore Data
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Step-1 - Load data
base_dir = os.getcwd()
file_path = os.path.join(base_dir, 'banking_complaints_2023.xlsx')  # Replace with actual filename

if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist.")
df = pd.read_excel(file_path)


# Display initial data summary
print(df.head())
print(f"\nDataset shape: {df.shape}")
print("\nMissing values per column:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)
print("\nDataset info:")
print(df.info())
print("\nSummary statistics:")
print(df.describe().transpose())

# Observations:
# - The dataset contains 7011 rows and 8 attributes with customer complaints. 
# - The 'Product' column contains categorical data, which may need to be encoded for modeling.
# - The 'Date Received' column is in string format and needs to be converted to datetime.
# - The 'Consumer complaint narrative' column contains text data that will be used for NLP tasks.
# - State and ZIP code columns may need to be cleaned or transformed for analysis due to potential missing values or formatting issues.

# Step -2 & 3 - Check types and date range
df['Date Received'] = pd.to_datetime(df['Date Received'], errors='coerce')
print(df.dtypes)
print("Date Range:", df['Date Received'].min(), "to", df['Date Received'].max())

# Observations:
# - The 'Date Received' column is in datetime format, which is suitable for time series analysis.

# Visualize complaints over time
plt.figure(figsize=(12,6))
df['Date Received'].value_counts().sort_index().plot()
plt.title('Complaint Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Complaints')
plt.grid(True)
plt.show()


# Step -4 - Data Preprocessing
# Check for missing values in 'Consumer complaint narrative' column
def preprocessing(text):
    text = text.lower() #Lowercase
    text = re.sub(r'\d+', '', text) #Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation)) # removes all punctuation characters
    tokens = text.split() # String to words
    stop_words = set(stopwords.words('english')) # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words] # Token
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Step -5 Apply preprocessing
df['clean_text'] = df['Complaint Description'].astype(str).apply(preprocessing)

# Observations:
# - The 'clean_text' column now contains preprocessed text data, which is ready for further analysis.
# - The preprocessing steps include converting to lowercase, removing digits, punctuation, stop words, and lemmatization.
# - The 'clean_text' column will be used for text classification and sentiment analysis.

# Step - 6 TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(df['clean_text'])

# Step -7 - Target variable for classification
y = df['Banking Product']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Observations:
# - The Random Forest Classifier achieved a good accuracy on the test set.
# - The classification report provides precision, recall, and F1-score for each class.
# - The model can be further tuned for better performance.

# Step - 8 - Transformer-based Classification(BERT Model)
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Example: Predict for a few complaints
sample_texts = df['Complaint Description'].head(5).tolist()
bert_preds = [classifier(text[:512])[0] for text in sample_texts]
print(bert_preds)

# Step -9 - Sentiment Analysis using VADER
analyzer = SentimentIntensityAnalyzer()
df['sentiment_scores'] = df['Complaint Description'].apply(lambda x: analyzer.polarity_scores(str(x)))
df['compound'] = df['sentiment_scores'].apply(lambda x: x['compound'])
df['sentiment'] = df['compound'].apply(lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral'))
print(df[['Complaint ID', 'compound', 'sentiment']].head())

# Step 10 - Business Insights
# Aggregate sentiment by product
sentiment_summary = df.groupby(['Banking Product', 'sentiment']).size().unstack().fillna(0)
print(sentiment_summary)

# Example Insight: If most complaints in "Credit Reporting" are negative, prioritize improvements there
# Visualize sentiment distribution
sentiment_summary.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Sentiment Distribution by Banking Product')
plt.xlabel('Banking Product')

plt.ylabel('Number of Complaints')
plt.legend(title='Sentiment')
plt.show()
# Observations:
# - The sentiment distribution plot shows the number of complaints categorized by sentiment for each banking product.
# - This visualization helps identify areas where customer sentiment is predominantly negative, indicating potential issues that need to be addressed.
# - The insights gained from this analysis can guide the bank in prioritizing improvements in specific areas to enhance customer satisfaction.
# - Most complaint is with Checking or Savings Account, thus need more attention
