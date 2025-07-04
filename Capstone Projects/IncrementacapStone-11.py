import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as plt
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# === Load & Inspect Dataset ===
#FileName
#fileName = "GrammarandProductReviews-Full.csv"
fileName = "GrammarandProductReviews.csv"
try:
    df = pd.read_csv(fileName, encoding='ISO-8859-1', on_bad_lines='skip')
    df.columns = df.columns.str.strip()
except FileNotFoundError:
    print(f"Error: The file '{fileName}' was not found. Please check the file path.")
    exit()
except pd.errors.ParserError as e:
    print(f"Error: Failed to parse the file '{fileName}'. Please check the file format. Details: {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit()

print("Dataset Info:")
df.info()
print("\nFirst 5 Rows:\n", df.head())
print("\nShape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

plt.hist(df['reviews.rating'], bins=10, color='blue', alpha=0.7)
plt.title('Distribution of Reviews Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()


# === Create Target Column ===
df['target'] = (df['reviews.rating'] < 4).astype(int)
print("\nTarget Distribution:\n", df['target'].value_counts(normalize=True))

plt.bar([0, 1], df['target'].value_counts(), color=['green', 'red'], alpha=0.7)
plt.title('Distribution of Target Classes')
plt.xlabel('Target Class')
plt.ylabel('Frequency')
plt.xticks([0, 1], ['Positive', 'Negative'])
plt.show()


# === Define Features and Labels ===
X = df['reviews.text'].astype(str)  # Ensure text data is string type
Y = df['target']

# === Train/Test Split (80/20) ===
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"\nTrain Size: {len(X_train)}, Test Size: {len(X_test)}")

#print Shapes of X_train and Y_train
print("\nX_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("\nX_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

# === Tokenization and Padding ===
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 150

# Vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, oov_token="<OOV>", char_level=False)
tokenizer.fit_on_texts(X_train)

# This will convert the text into sequences of integers, where each integer represents a word in the vocabulary

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Word index is a dictionary mapping words to their index in the tokenizer's vocabulary
word_index = tokenizer.word_index
print("\nWord Index Size:", len(word_index))
print("\nSample Word Index:\n", dict(list(word_index.items())[:10]))

index_to_word = dict((i, word) for word, i in word_index.items())
print("\nSample Index to Word Mapping:\n", dict(list(index_to_word.items())[:10]))
" ".join([index_to_word[i] for i in range(1, 11)])
print("\nSample Text from Index 1 to 10:\n", " ".join([index_to_word[i] for i in range(1, 11)]))

# The histogram plotted below shows the distribution of the number of words in each review
plt.hist([len(x) for x in X_train_seq], bins=50, color='blue', alpha=0.7)
plt.title('Distribution of Number of Words in Reviews')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.show()



X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH)
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH)

print("\nX_train_pad shape:", X_train_pad.shape)
print("X_test_pad shape:", X_test_pad.shape)

# === Word Index Sample ===
word_index = tokenizer.word_index
print("\nSample Word Index:\n", dict(list(word_index.items())[:10]))

# === One-hot Encoding Targets ===
Y_train_oh = tf.keras.utils.to_categorical(Y_train, num_classes=2)
Y_test_oh = tf.keras.utils.to_categorical(Y_test, num_classes=2)

print("\nY_train_oh shape:", Y_train_oh.shape)
print("Y_test_oh shape:", Y_test_oh.shape)


# === Build CNN-LSTM Hybrid Model ===
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'),
    tf.keras.layers.Embedding(input_dim=MAX_NB_WORDS, output_dim=50, input_length=MAX_SEQUENCE_LENGTH),
    tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=5),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=5),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(2, activation='softmax')
])

# === Compile the Model ===
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === Train the Model ===
history = model.fit(X_train_pad, Y_train_oh, epochs=5, batch_size=64, validation_split=0.2)

# === Evaluate the Model ===
loss, accuracy = model.evaluate(X_test_pad, Y_test_oh, verbose=0)
print(f"\nTest Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

