#Course 04: Course End Project-Project-1
'''
Create a model that predicts whether or not a loan will be default using historical data
'''

# Perform data preprocessing and build a deep learning prediction model
# Import necessary libraries
# Import necessary libraries (remove unused imports)
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
import warnings
warnings.filterwarnings("ignore")



# Load the dataset
base_dir = os.getcwd()
fileName = os.path.join(base_dir, 'Project-2Data', 'loan_data.csv')

# Check if the file exists before reading
if os.path.exists(fileName):
	df = pd.read_csv(fileName)
else:
	raise FileNotFoundError(f"The file {fileName} does not exist. Please check the path.")

# Display the first few rows of the dataset
print(df.head(10))

# Display shape and info of the dataset
print(f"Shape of the dataset: {df.shape}")
print(f"Columns in the dataset: {df.columns}")
print(f"Data types of the columns: {df.dtypes}")
# Check for missing values
print("Missing values in each column:")
if df.isnull().sum().any():
    print("There are missing values in the dataset:")
    print(df.isnull().sum())
else:
    print("No missing values in the dataset.")
print(df.info())
print(df.describe())

# Check for class imbalance
print("Class distribution:")
print(df['credit.policy'].value_counts(normalize=True))  # Check for class imbalance
# Visualize class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='credit.policy', data=df)
plt.title('Class Distribution of Loan Status')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.show()

# Encode Categorical Features
df = pd.get_dummies(df, columns=['purpose'], drop_first=True) # Encode the purpose feature using one-hot encoding.

#  Step 3: Feature Engineering

# Correlation
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.show()

# Drop features with high correlation (e.g., correlation coefficient > 0.9).
#  Step 4: Define Target and Features
X = df.drop('credit.policy', axis=1)  # Features
y = df['credit.policy']

# Step 5: Train/Test Split and Scaling

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#  Step 6: Build a Deep Learning Model (Keras)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Step 7: Evaluate the Model

preds = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, preds))


print("Confusion Matrix:")
print(confusion_matrix(y_test, preds))
print("Accuracy Score:")
print(accuracy_score(y_test, preds))

# Hnadle Class Imbalance

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Fit the model again with the resampled data
model.fit(
    X_train_res, y_train_res,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)
# Save the model
model.save("lending_club_default_model.h5")

# Load the model
model = load_model("lending_club_default_model.h5")


# Predict on the test set again
preds = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test, preds))
print("Accuracy Score:")
print(accuracy_score(y_test, preds))

# Visualize the confusion matrix
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'])    
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Correctly calculate and interpret sensitivity and specificity
TN, FP, FN, TP = cm.ravel()
sensitivity = TP / (TP + FN)  # True Positive Rate
specificity = TN / (TN + FP)  # True Negative Rate
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")

# Plot AUC-ROC Curve

y_pred_proba = model.predict(X_test).ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Calculate AUC-ROC Score
roc_auc_score_value = roc_auc_score(y_test, y_pred_proba)
print(f"AUC-ROC Score: {roc_auc_score_value:.4f}")
# Plot ROC Curve using sklearn's RocCurveDisplay
RocCurveDisplay.from_predictions(y_test, y_pred_proba, name='ROC Curve')
plt.title('ROC Curve using RocCurveDisplay')
plt.show()
