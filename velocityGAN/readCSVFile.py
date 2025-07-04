import pandas as pd

# Load CSV  Number of rows: 65818000, Number of columns: 36
df = pd.read_csv("D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/submission-GAN.csv")

# # Load CSV  Number of rows: 4607260, Number of columns: 36
# df = pd.read_csv("D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/submission-new.csv")

# Display the first 10 rows with header
print(df.head(10))

# Print tail of the DataFrame
print(df.tail(10))

print(df.shape)
# Print the columns of the DataFrame
print(df.columns)
# Print the data types of the columns
print(df.dtypes)
# Print the number of rows and columns
print(f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")