'''
Develop an end-to-end NLP pipeline to analyze customer reviews for sentiment classification and key topic extraction. 
This system will help BikeEase identify customer pain points and areas of improvement.

Task 1: Data Collection & preprocessing

Collect and clean customer reviews from a given dataset (or scrape data if available)
Perform text cleaning (lowercasing, removing punctuation, stopword removal, lemmatization)
Tokenize and vectorize the text if required
Task 2: Sentiment analysis

Build a sentiment classification model (positive, neutral, negative) using:
Traditional models: Logistic Regression, Naïve Bayes
Deep learning models: LSTMs, Transformers (BERT)
Evaluate models using accuracy, F1-score, and confusion matrix

'''

#Load libraries
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from transformers import BertTokenizer, TFBertModel
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, confusion_matrix
import os
import json
import requests
import logging

# Load Data


base_dir = os.getcwd()
file_path = os.path.join(base_dir, 'bike_rental_reviews.csv')

if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist.")

df = pd.read_csv(file_path)


# Display initial data summary
print(df.head())
print(f"\nDataset shape: {df.shape}")
print("\nMissing values per column:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)
print("\nDataset info:")
print(df.info())
print("\nSummary statistics:")
print(df.describe())

# Perform text cleaning (lowercasing, removing punctuation, stopword removal, lemmatization)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('treebank')
nltk.download('punkt_tab')


# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply text cleaning
if 'review_text' not in df.columns:
    raise KeyError("The dataset does not contain a 'review_text' column. Please check the dataset.")
df['cleaned_reviews'] = df['review_text'].apply(clean_text)

# Display cleaned text
print("\nCleaned reviews:")
print(df['cleaned_reviews'].head())
# Check for class imbalance
print("\nClass distribution:")
print(df['sentiment'].value_counts(normalize=True))  # Check for class imbalance

# Visualize class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='sentiment', data=df, palette='viridis')
plt.title('Class Distribution of Sentiment')

plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

#Tokenize and vectorize the text if required 
# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the cleaned reviews
X = vectorizer.fit_transform(df['cleaned_reviews']).toarray()
y = df['sentiment']

# Encode sentiment labels as integers
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['sentiment'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shape of the training and testing sets
print(f"\nTraining set shape: {X_train.shape}, Testing set shape: {X_test.shape}")

# Display the first few rows of the training set
print("\nFirst few rows of the training set:")
print(X_train[:5])
print(y_train[:5])


# Traditional Models: Logistic Regression and Naïve Bayes
# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
log_reg_f1 = f1_score(y_test, y_pred_log_reg, average='weighted')
log_reg_cm = confusion_matrix(y_test, y_pred_log_reg)

print("\nLogistic Regression Accuracy:", log_reg_accuracy)
print("Logistic Regression F1 Score:", log_reg_f1)
print("Logistic Regression Confusion Matrix:\n", log_reg_cm)
print(classification_report(y_test, y_pred_log_reg))

# Naïve Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
nb_accuracy = accuracy_score(y_test, y_pred_nb)
nb_f1 = f1_score(y_test, y_pred_nb, average='weighted')
nb_cm = confusion_matrix(y_test, y_pred_nb)

print("\nNaïve Bayes Accuracy:", nb_accuracy)
print("Naïve Bayes F1 Score:", nb_f1)
print("Naïve Bayes Confusion Matrix:\n", nb_cm)
print(classification_report(y_test, y_pred_nb))

# Plot confusion matrices for both models
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(log_reg_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.subplot(1, 2, 2)
sns.heatmap(nb_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.title('Naïve Bayes Confusion Matrix')
plt.xlabel('Predicted')

plt.ylabel('True')
plt.tight_layout()
plt.show()
# Deep Learning Models: LSTM and Transformers (BERT)
# LSTM Model
max_words = 5000
max_len = 100
embedding_dim = 128

# Tokenize the text data
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['cleaned_reviews'])
sequences = tokenizer.texts_to_sequences(df['cleaned_reviews'])
word_index = tokenizer.word_index
print(f"Found {len(word_index)} unique tokens.")

# Pad sequences to ensure uniform input size
X = pad_sequences(sequences, maxlen=max_len)
# Encode sentiment labels as integers
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['sentiment'])
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Display the shape of the training and testing sets
print(f"\nTraining set shape: {X_train.shape}, Testing set shape: {X_test.shape}")
# Display the first few rows of the training set
print("\nFirst few rows of the training set:")
print(X_train[:5])
print(y_train[:5])
# Build the LSTM model
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)
batch_size = 64
epochs = 5
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=2)
# Evaluate the model
y_pred_lstm = model.predict(X_test)
y_pred_lstm = np.argmax(y_pred_lstm, axis=1)
lstm_accuracy = accuracy_score(y_test, y_pred_lstm)
lstm_f1 = f1_score(y_test, y_pred_lstm, average='weighted')
lstm_cm = confusion_matrix(y_test, y_pred_lstm)

print("\nLSTM Model Accuracy:", lstm_accuracy)
print("LSTM Model F1 Score:", lstm_f1)
print("LSTM Model Confusion Matrix:\n", lstm_cm)
print(classification_report(y_test, y_pred_lstm))
# Plot LSTM training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('LSTM Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()
# BERT Model
# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Tokenize and encode the text data
train_encodings = tokenizer(list(df['cleaned_reviews']), truncation=True, padding=True, max_length=128, return_tensors='tf')
test_encodings = tokenizer(list(df['cleaned_reviews']), truncation=True, padding=True, max_length=128, return_tensors='tf')

# Convert labels to tensors
train_labels = tf.convert_to_tensor(y_train)
test_labels = tf.convert_to_tensor(y_test)
# Create a TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), test_labels)).batch(32)
# Build the BERT model
model = Sequential()
model.add(bert_model)
model.add(Dense(3, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train the BERT model
history = model.fit(train_dataset, epochs=3, validation_data=test_dataset)
# Evaluate the BERT model
y_pred_bert = model.predict(test_dataset)
y_pred_bert = np.argmax(y_pred_bert, axis=1)
bert_accuracy = accuracy_score(y_test, y_pred_bert)
bert_f1 = f1_score(y_test, y_pred_bert, average='weighted')
bert_cm = confusion_matrix(y_test, y_pred_bert)

print("\nBERT Model Accuracy:", bert_accuracy)
print("BERT Model F1 Score:", bert_f1)
print("BERT Model Confusion Matrix:\n", bert_cm)
print(classification_report(y_test, y_pred_bert))
# Plot BERT training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('BERT Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('BERT Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()





