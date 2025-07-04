import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, f1_score, RocCurveDisplay, 
                             ConfusionMatrixDisplay)

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "8"  # Adjust based on your CPU

# Task-2 Load dataset
df = pd.read_csv("adultcensusincome.csv", encoding='ISO-8859-1')
df.columns = df.columns.str.strip()

# Task-3 Data Inspection
df.info()
print("\nFirst 5 Rows:\n", df.head())
print("\nSummary Statistics:\n", df.describe().transpose())  
print("\nMissing Values:\n", df.isnull().sum())

# Task 4: Identify missing values
print("\nMissing Values:\n", df.isnull().sum())
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces in column names
# Checking Dataset that ? is present in the dataset
print(df.isin(['?']).sum())

income = round(df['income'].value_counts(normalize=True) * 100, 2).astype(str) + '%'
print(income)

# Data Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
sns.countplot(x='income', data=df, ax=axes[0, 0]).set_title('Income Distribution')
sns.histplot(df['age'], kde=True, color='blue', ax=axes[0, 1]).set_title('Age Distribution')
sns.countplot(y='education', data=df, order=df['education'].value_counts().index, ax=axes[0, 2])
axes[0, 2].set_title('Education Distribution')
df['marital.status'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90,
    colors=sns.color_palette('pastel'), ax=axes[1, 0])
axes[1, 0].set_title('Marital Status Distribution')
axes[1, 0].set_ylabel('')
sns.countplot(y='education.num', data=df, ax=axes[1, 1]).set_title('Years of Education Distribution')
plt.tight_layout()
plt.show()


# Correlation heatmap
le = LabelEncoder()
df['income'] = le.fit_transform(df['income'])

#Replace the '?' with NaN
df.replace('?', np.nan, inplace=True)
df_numeric = df.select_dtypes(include=['number'])
corr = df_numeric.corr()

with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(12, 8))
    ax = sns.heatmap(corr, annot=True, cmap='rocket', fmt=".2f", mask=False, square=True, linewidths=0.5)
    plt.title('Correlation Heatmap of Numeric Columns')
    plt.show()


#Colum with Null value
# Check for missing values in the dataset
missing_values = df.isnull().sum()
columns_with_nan =['workclass', 'occupation', 'native.country']
# Fill missing values in the 'workclass', 'occupation', and 'native.country' columns with the mode
for col in columns_with_nan:
    df[col].fillna(df[col].mode()[0], inplace=True)

#Print the missing values in the dataset
print("\nMissing Values after replacing '?' with mode:\n", df.isnull().sum())


# Label Encoding
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Feature Scaling & Handling Imbalance
X = df.drop('income', axis=1)
y = df['income']
X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
X_resampled, y_resampled = SMOTE(random_state=1).fit_resample(X_scaled, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Model Training
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average='weighted'),
        "y_pred": y_pred
    }
    print(f"\n{name} Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrices & ROC Curves
for name, res in results.items():
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, res["y_pred"]), display_labels=['<=50K', '>50K'])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.show()
    RocCurveDisplay.from_estimator(models[name], X_test, y_test)
    plt.title(f'ROC Curve - {name}')
    plt.show()

# Identify Best Model
best_model = max(results, key=lambda x: results[x]["f1_score"])
print(f"\nBest Model: {best_model} (F1 Score: {results[best_model]['f1_score']:.4f})")

# Hyperparameter Tuning
param_grids = {
    "Random Forest": {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},
    "Decision Tree": {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},
    "SVM": {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    "KNN": {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']},
    "Logistic Regression": {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
}

if best_model in param_grids:
    grid_search = GridSearchCV(models[best_model], param_grids[best_model], cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_tuned_model = grid_search.best_estimator_
    y_pred_tuned = best_tuned_model.predict(X_test)
    tuned_f1 = f1_score(y_test, y_pred_tuned, average='weighted')
    print(f"Best Parameters for {best_model}: {best_params}")
    print(f"Tuned F1 Score for {best_model}: {tuned_f1:.4f}")

# # Display ConfusionMatrix side by side after tuning with the original model
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))


cm_before = confusion_matrix(y_test, results[best_model]["y_pred"])
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_before, display_labels=['<=50K', '>50K'])
ax1.set_title(f'Confusion Matrix:Before Tuning - {best_model}')
ax1.grid(False)
disp1.plot(cmap='Blues', ax=ax1)

cm_after = confusion_matrix(y_test, y_pred_tuned)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_after, display_labels=['<=50K', '>50K'])
ax2.set_title(f'Confusion Matrix: After - {best_model} Tuned')
ax2.grid(False)
disp2.plot(cmap='Blues', ax=ax2)
plt.tight_layout()
plt.show()

print('Before Tuning')
print(classification_report(y_test, results[best_model]["y_pred"]))
print('After Tuning')
print(classification_report(y_test, y_pred_tuned))