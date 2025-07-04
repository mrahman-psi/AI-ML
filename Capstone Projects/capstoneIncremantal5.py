# Task 1: Import relevant Python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split    

# Task 2: Load the dataset with appropriate encoding
df = pd.read_csv("FloridaBikeRentals.csv", encoding='ISO-8859-1')

# Task 3: Inspect the dataset
df.info()
print("\nFirst 5 Rows:")
print(df.head())
print("\nLast 5 Rows:")
print(df.tail())
print("\nSummary Statistics:")
print(df.describe().transpose)

# Task 4: Identify missing values
print("\nMissing Values:")
print(df.isnull().sum())
df.columns = df.columns.str.strip()

# Task 5: Convert 'Date' column to datetime format and extract new features
df = df.dropna(subset=['Date'])
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek

# Mark weekends: Saturday (5) or Sunday (6) will be marked as 1, others as 0
df['Weekend'] = (df['DayOfWeek'] >= 5).astype(int)

print("\nFirst 5 Rows after adding Day, Month, DayOfWeek, and Weekend:")
print(df.head())

# Check correlation between features using heatmap
plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

#Plot the distribution plot of Rented Bike count
plt.figure(figsize=(10, 6)) 
sns.distplot(df['Rented Bike Count'], color='blue')
plt.title('Rented Bike Count Distribution')
plt.show()

#Plot the histogram of all numeric features
plt.figure(figsize=(10, 6))
df.hist(bins=20, color='blue', edgecolor='black', grid=False, figsize=(12,8))
plt.tight_layout()
plt.show()

#Plot the box plot of the Rented Bike Count against all the categorical features (Categorical featues on X-axis and Rentend Bike Count on Y-axis)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Seasons', y='Rented Bike Count', data=df)
plt.title('Rented Bike Count vs Seasons')
plt.show()

#Plot the Seabon catplot of Rented Bike Count against features 'Hour', Holiday, Rainfall, Snowfall and Weekdays, Weekend and give your inference
plt.figure(figsize=(10, 6))
sns.catplot(x='Hour', y='Rented Bike Count', hue='Holiday', data=df, kind='bar')
plt.title('Rented Bike Count vs Hour with Holiday')
#plt.savefig("sine_wave.pdf")
plt.show()

# Ploting Catergorical features against Rented Bike Count
feature_list = ['Hour', 'Seasons', 'Holiday', 'Rainfall(mm)', 'Snowfall (cm)', 'Weekend']
for feature in feature_list:
    plt.figure(figsize=(10, 6))
    sns.catplot(x=feature, y='Rented Bike Count', data=df, kind='bar')
    plt.title(f'Rented Bike Count vs {feature}')
    plt.show()  
    
#Encode the Categorical features into the numerical features
categorical_columns = ['Seasons', 'Holiday', 'Functioning Day']
df_dummies = pd.get_dummies(df, columns=categorical_columns)
df_dummies.to_csv("Rental_Bike_Data_Dummy.csv", index=False)
print("Data exported successfully as Rental_Bike_Data_Dummy.csv")

#Identify the target variable and split the dataset into train and test with a ratio of 80:20 and random state 1
X = df_dummies.drop(['Rented Bike Count', 'Date'], axis=1)
y = df_dummies['Rented Bike Count']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)    

#Perform Standard Scaling on the train dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Perform Linear Regression on the train dataset
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_scaled, y_train)
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)
print(f"LinearRegression Train Score: {train_score}")
print(f"LinearRegression Test Score: {test_score}")

#Perform Lasso Regression on the train dataset
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.01)
model.fit(X_train_scaled, y_train)
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)
print(f"Lasso Train Score: {train_score}")
print(f"Lasso Test Score: {test_score}")

#Perform Ridge Regression on the train dataset
from sklearn.linear_model import Ridge
model = Ridge(alpha=0.01)
model.fit(X_train_scaled, y_train)
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)
print(f"Ridge Train Score: {train_score}")
print(f"Ridge Test Score: {test_score}")





