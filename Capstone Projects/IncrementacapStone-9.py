# === Import Libraries ===
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# === Load & Inspect Dataset ===
df = pd.read_csv("Churn_Modelling.csv", encoding='ISO-8859-1')
df.columns = df.columns.str.strip()

print("Dataset Info:")
df.info()
print("\nFirst 5 Rows:\n", df.head())
print("\nShape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# === Preprocessing ===
def preprocess_data(df):
    df = df.drop(columns=['CustomerId', 'Surname', 'CreditScore'])

    # Encode 'Gender'
    gender_encoder = LabelEncoder()
    df['Gender'] = gender_encoder.fit_transform(df['Gender'])

    # One-hot encode 'Geography'
    geo_encoder = OneHotEncoder(drop='first', sparse_output=False)
    geo_encoded = geo_encoder.fit_transform(df[['Geography']])
    geo_encoded_df = pd.DataFrame(geo_encoded, 
                                   columns=geo_encoder.get_feature_names_out(['Geography']), 
                                   index=df.index)
    df = pd.concat([df.drop(columns='Geography'), geo_encoded_df], axis=1)

    return df, gender_encoder, geo_encoder

df, gender_encoder, geo_encoder = preprocess_data(df)

# Split features and target
X = df.drop(columns='Exited')
Y = df['Exited']

# Fill missing values (if any)
X.fillna(0, inplace=True)
Y.fillna(0, inplace=True)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Train shape:", X_train.shape, Y_train.shape)
print("Test shape:", X_test.shape, Y_test.shape)

# === Build & Train Neural Network ===
model = Sequential([
    Dense(6, activation='relu', input_dim=X_train.shape[1]),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, Y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=10,
    callbacks=[early_stop],
    verbose=1
)

# === Evaluation ===
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Loss over Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title("Accuracy over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_history(history)

# Predictions
y_pred = model.predict(X_test) > 0.5
cm = confusion_matrix(Y_test, y_pred)

# Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Metrics
print("\nClassification Report:\n", classification_report(Y_test, y_pred))
print("\nMetrics:")
print(f"Accuracy: {accuracy_score(Y_test, y_pred) * 100:.2f}%")
print(f"Precision: {precision_score(Y_test, y_pred) * 100:.2f}%")
print(f"Recall: {recall_score(Y_test, y_pred) * 100:.2f}%")
print(f"F1 Score: {f1_score(Y_test, y_pred) * 100:.2f}%")

# === Predict for a New Customer ===
def predict_new_customer(customer_dict, gender_encoder, geo_encoder, scaler, model, feature_columns):
    df_new = pd.DataFrame([customer_dict])

    # Gender encoding
    df_new['Gender'] = gender_encoder.transform(df_new['Gender'])

    # Geography encoding with handling unknowns
    known_geos = geo_encoder.categories_[0]
    df_new['Geography'] = df_new['Geography'].apply(lambda g: g if g in known_geos else known_geos[0])
    geo_encoded = geo_encoder.transform(df_new[['Geography']])
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out(['Geography']), index=df_new.index)

    df_new = pd.concat([df_new.drop(columns='Geography'), geo_encoded_df], axis=1)

    # Ensure column alignment
    for col in feature_columns:
        if col not in df_new.columns:
            df_new[col] = 0
    df_new = df_new[feature_columns]

    # Scale and predict
    df_scaled = scaler.transform(df_new)
    pred = model.predict(df_scaled)[0][0]
    return "Leave" if pred > 0.5 else "Stay"

new_customer = {
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000,
    'CreditScore': 600  # Will be ignored (no longer used)
}

prediction = predict_new_customer(new_customer, gender_encoder, geo_encoder, scaler, model, X.columns)
print(f"\nPrediction for the new customer: The customer is predicted to {prediction}.")
