import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Set max CPU count for parallel processing
os.environ["LOKY_MAX_CPU_COUNT"] = "8"

def load_data(file_path):
    """Loads dataset and validates file existence."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found. Check the file path.")
    return pd.read_csv(file_path, encoding='ISO-8859-1')

def check_missing_values(df):
    """Checks for missing values in the dataset."""
    missing = df.isnull().sum()
    print("\nMissing Values:")
    print(missing[missing > 0] if missing.any() else "None")

def plot_turnover_by_department(df):
    """Visualizes employee turnover count and percentage by department."""
    turnover = df.groupby(['sales', 'left']).size().unstack()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    turnover.plot(kind='bar', stacked=True, ax=axes[0], colormap='coolwarm')
    axes[0].set_title('Employee Turnover by Department')
    
    (turnover.div(turnover.sum(axis=1), axis=0) * 100).plot(kind='bar', stacked=True, ax=axes[1], colormap='viridis')
    axes[1].set_title('Turnover Percentage by Department')
    
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df):
    """Displays correlation heatmap."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.show()

def plot_distributions(df):
    """Plots distributions of key features."""
    features = ['satisfaction_level', 'last_evaluation', 'average_montly_hours']
    df[features].hist(bins=30, figsize=(12, 5), layout=(1, 3), color='blue', alpha=0.7)
    plt.show()

def plot_project_count_by_turnover(df):
    """Plots project count for employees who left vs stayed."""
    plt.figure(figsize=(10, 6))
    sns.countplot(x='number_project', hue='left', data=df, palette='Set2')
    plt.title('Project Count by Turnover')
    plt.tight_layout()
    plt.show()

def cluster_employees_who_left(df):
    """Clusters employees who left based on satisfaction and evaluation."""
    data = df[df['left'] == 1][['satisfaction_level', 'last_evaluation']].copy()
    scaled_data = StandardScaler().fit_transform(data)
    
    kmeans = KMeans(n_clusters=3, random_state=123, n_init=10)
    data['cluster'] = kmeans.fit_predict(scaled_data)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='satisfaction_level', y='last_evaluation', hue='cluster', palette='viridis', s=100)
    plt.title('Clusters of Employees Who Left')
    plt.show()

def handle_class_imbalance(df):
    """Handles class imbalance using SMOTE."""
    X = pd.get_dummies(df.drop('left', axis=1), drop_first=True)
    y = df['left']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    return X_train_res, X_test, y_train_res, y_test

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    """Trains a model and evaluates performance."""
    model = RandomForestClassifier(random_state=123, n_estimators=100)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return model

def plot_roc_auc(model, X_test, y_test):
    """Plots ROC curve and calculates AUC."""
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='red')
    plt.legend()
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    plt.show()

def main():
    file_path = "ClassProject/HR_comma_sep.csv"
    df = load_data(file_path)
    check_missing_values(df)
    plot_turnover_by_department(df)
    plot_correlation_heatmap(df)
    plot_distributions(df)
    plot_project_count_by_turnover(df)
    cluster_employees_who_left(df)
    
    X_train, X_test, y_train, y_test = handle_class_imbalance(df)
    model = train_and_evaluate_model(X_train, y_train, X_test, y_test)
    plot_roc_auc(model, X_test, y_test)

if __name__ == "__main__":
    main()
