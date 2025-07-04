# Task 1: Import relevant Python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "8"  # Adjust based on your CPU

# Task 2: Load the dataset with appropriate encoding
df = pd.read_csv("adultcensusincome.csv", encoding='ISO-8859-1')

# Task 3: Inspect the dataset
df.info()
print("\nFirst 5 Rows:\n", df.head())
print("\nLast 5 Rows:\n", df.tail())
print("\nSummary Statistics:\n", df.describe().transpose())  

# Task 4: Identify missing values
print("\nMissing Values:\n", df.isnull().sum())
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces in column names

# Checking Dataset that ? is present in the dataset
print(df.isin(['?']).sum())

income = round(df['income'].value_counts(normalize=True) * 100, 2).astype(str) + '%'
print(income)

# Create visualizations Bar plot for income distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='income', data=df, palette='bright')
plt.title('Income Distribution')
plt.xlabel('Income')
plt.ylabel('Number of People')
plt.tick_params(axis='x', labelsize=10)
plt.show()

# Alternative way to plot the income distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=df['income'].value_counts().index, y=df['income'].value_counts().values, palette='bright')
plt.title('Income Distribution - Bar Plot')
plt.xlabel('Income')
plt.ylabel('Number of People')
plt.tick_params(axis='x', labelsize=10)
plt.show()


#Create a distribution plot for the 'age' column
plt.figure(figsize=(10, 6))
sns.set_style('whitegrid')
sns.histplot(df['age'], kde=True, color='blue', bins=20)  
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Number of People')
plt.tick_params(labelsize=10)
plt.show()

#Create a Bar plot for the 'education' column
edu=df['education'].value_counts()
plt.figure(figsize=(10, 6))
plt.style.use('fivethirtyeight')
sns.barplot(x=edu.values, y=edu.index, palette='Paired')
plt.title('Education Distribution', fontdict={'fontname':'Monospace', 'fontsize': 15, 'fontweight': 'bold'})
plt.xlabel('Number of People')
plt.ylabel('Education')
plt.tick_params(labelsize=10)
plt.show()

# Create a Bar Plot for Years of Education Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='education.num', data=df, palette='colorblind')
plt.title('Years of Education Distribution', fontdict={'fontname':'Monospace', 'fontsize': 15, 'fontweight': 'bold'})
plt.xlabel('Years of Education')
plt.ylabel('Number of People')
plt.tick_params(labelsize=10)
plt.show()


#Create a pie chart for the 'marital.status' column
# Improved Pie Chart for Marital Status
plt.figure(figsize=(10, 8))
plt.style.use('ggplot')
marital_status_counts = df['marital.status'].dropna().value_counts()
labels = marital_status_counts.index  # Labels for legend
explode = [0.1] + [0] * (len(marital_status_counts) - 1)  # Dynamically set explode
marital_status_counts.plot.pie(autopct='%1.1f%%', startangle=10, shadow=True, explode=explode, 
                               colors=sns.color_palette('pastel'), labels=['']*len(labels))  # Hide default labels
plt.title('Marital Status Distribution', fontdict={'fontname':'Monospace', 'fontsize': 15, 'fontweight': 'bold'})
plt.legend(labels, title='Marital Status', bbox_to_anchor=(1.2, 1), prop={'size': 7}, loc='upper right')  
plt.axis('equal')  # Ensures pie chart remains circular
plt.show()


# Create Bar Plot for Column Sex
sex = df['sex'].value_counts()
plt.figure(figsize=(10, 5))
plt.style.use('fivethirtyeight')
sns.barplot(x=sex.index, y=sex.values, palette='viridis')
plt.title('Distribution of Sex', fontdict={'fontname':'Monospace', 'fontsize': 15, 'fontweight': 'bold'})
plt.xlabel('Sex')
plt.ylabel('Number of People')

# Create a Bar Plot for column Hours per Week
hours=df['hours.per.week'].value_counts().head(10)
plt.style.use('fivethirtyeight')
plt.figure(figsize=(15, 7))
sns.barplot(x=hours.index, y=hours.values, palette='colorblind')
plt.title('Hours per Week Distribution', fontdict={'fontname':'Monospace', 'fontsize': 15, 'fontweight': 'bold'})
plt.xlabel('Working hours per Week')
plt.ylabel('Number of People')
plt.show()

#Bi-variate Analysis
# Create a bar plot for income distribution across different categories
categorical_vars = ['age', 'education', 'marital.status', 'race', 'sex']
for var in categorical_vars:
    plt.figure(figsize=(12, 6))
    sns.countplot(x=var, hue='income', data=df, order=df[var].value_counts().index)
    plt.title(f'Income Distribution across {var}')
    plt.xticks(rotation=45)
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



# # # Label encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


# Prepare independent variables (X) and dependent variable (y)
X = df.drop('income', axis=1)
y = df['income']

#Perform Feature Scaling using StandardScaler and fix the imbalance using RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
print(y_resampled.value_counts(normalize=True))

# # Train-test split (80:20 ratio)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# Perform feature scaling and handle imbalance using SMOTE
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)
smote = SMOTE(random_state=1)
X_smote, y_smote = smote.fit_resample(X, y)

# Train-test split (80:20 ratio)
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

# Train models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(random_state=42),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

y_preds = {}
accuracy_scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_preds[name] = model.predict(X_test)
    accuracy_scores[name] = accuracy_score(y_test, y_preds[name])

# Print accuracy of models
for name, acc in accuracy_scores.items():
    print(f"{name} Accuracy: {acc:.4f}")

# Calculate and print F1 scores
f1_scores = {name: f1_score(y_test, y_preds[name], average='weighted') for name in models}
for name, f1 in f1_scores.items():
    print(f"F1 Score - {name}: {f1:.4f}")

# Identify the best model based on F1 score
best_model = max(f1_scores, key=f1_scores.get)
print(f"\nBest Model based on F1 Score: {best_model}")

# Generate classification reports
for name, y_pred in y_preds.items():
    print(f"\n{name} Classification Report:\n")
    print(classification_report(y_test, y_pred))

# Plot confusion matrices
for name, model in models.items():
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_preds[name])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['<=50K', '>50K'])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.show()

# Plot ROC curves
for name, model in models.items():
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f'ROC Curve - {name}')
    plt.show()



# Tune hyperparameters of the best model
# Perform GridSearchCV to find the best hyperparameters for the best model

# Print Inital Random Forest Parameters
print(f"Best Parameter: {models[best_model].get_params() if best_model in models else 'Model not found'}")  # Print initial parameters of the best model


# Define hyperparameter grid based on the best model
if best_model == "Random Forest":
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    model = RandomForestClassifier()

elif best_model == "Decision Tree":
    param_grid = {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    model = DecisionTreeClassifier()

elif best_model == "SVM":
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }
    model = SVC()

elif best_model == "KNN":
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    }
    model = KNeighborsClassifier()

elif best_model == "Logistic Regression":
    param_grid = {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    }
    model = LogisticRegression()

else:
    param_grid = {}
    model = models[best_model]

# Perform GridSearchCV
if param_grid:
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_model_tuned = grid_search.best_estimator_
    print(f"Best hyperparameters for {best_model}: {best_params}")
    
    # Evaluate tuned model
    y_pred_tuned = best_model_tuned.predict(X_test)
    tuned_f1 = f1_score(y_test, y_pred_tuned, average='weighted')
    print(f"Tuned F1 Score for {best_model}: {tuned_f1:.4f}")
else:
    print("No hyperparameter tuning required for this model.")

# # Display ConfusionMatrix side by side after tuning with the original model
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))


# cm_before = confusion_matrix(y_test, y_preds[best_model])
# disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_before, display_labels=['<=50K', '>50K'])
# ax1.set_title(f'Confusion Matrix - {best_model}')
# ax1.grid(False)
# disp1.plot(cmap='Blues', ax=ax1)

# cm_after = confusion_matrix(y_test, y_pred_tuned)
# disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_after, display_labels=['<=50K', '>50K'])
# ax2.set_title(f'Confusion Matrix - {best_model} Tuned')
# ax2.grid(False)
# disp2.plot(cmap='Blues', ax=ax2)
# plt.tight_layout()
# plt.show()

# print('Before Tuning')
# print(classification_report(y_test, y_preds[best_model]))
# print('After Tuning')
# print(classification_report(y_test, y_pred_tuned))







