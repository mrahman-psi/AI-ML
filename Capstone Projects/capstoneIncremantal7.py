import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Adjust based on your CPU


# Load dataset
df = pd.read_csv("CC-GENERAL.csv", encoding='ISO-8859-1')
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces

# Inspect dataset
df.info()
print("Rows and Columns:", df.shape)
print("Columns:", df.columns[1])
print("\nFirst 5 Rows:\n", df.head())
print("\nSummary Statistics:\n", df.describe().transpose())
print("\nMissing Values:\n", df.isnull().sum())

# Handle missing values with zero as minimum payment zero makes more sense for the context as well for the one credit limit
# and one minimum payment column
if df.isnull().sum().sum() > 0:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    print("\nMissing Values after filling:\n", df.isnull().sum())
else:
    print("\nNo missing values in the dataset.")


numeric_cols = df.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[numeric_cols])
norm_data = normalize(X_scaled)


# Perform PCA and plot explained variance
pca = PCA()
X_pca = pca.fit_transform(norm_data)
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratios:\n", explained_variance)

cumulative_explained_variance = np.cumsum(explained_variance)

# Plot explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.title('PCA Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.show()

# Perform PCA with 2 components for visualization
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(norm_data)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca_2[:, 0], y=X_pca_2[:, 1], palette='viridis', alpha=0.6)
plt.title("PCA Visualization (2 Components)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title='TENURE', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.show()


# # Find the 2 columns which give the most covariances.
# cov_matrix = np.cov(norm_data.T)
# plt.figure(figsize=(10, 6))
# sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', fmt=".2f", mask=np.triu(np.ones_like(cov_matrix, dtype=bool)), linewidths=0.5)
# plt.title('Covariance Matrix')
# plt.show()

# Perform PCA with 2 principal components with the aim of visualizing clustering.
#Here is how to determine the feature from the original dataset
for i in np.arange(2):
    index = np.argmax(np.absolute(pca.get_covariance()[i]))
    max_cov = pca.get_covariance()[i][index]
    column = numeric_cols[index]
    print("Principal Component", i+1, "maximum covariance :", "{:.2f}".format(max_cov), "from column", column)


#Kmeans clustering
sse ={}
n_clust = np.arange(2, 11)
for k in n_clust:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca_2)
    sse[k] = kmeans.inertia_

plt.figure(figsize=(10, 6))
plt.plot(list(sse.keys()), list(sse.values()), marker='o', linestyle='--')
plt.xlabel("Number of Clusters")
plt.ylabel("Within cluster sum of squares")
plt.title("KMeans Clustering")
plt.grid()
plt.show()


# Perform K Means Clustering on the 2 component PCA transformed data with clusters ranging from 2 to 11 and plot the K Means inertia against the number of clusters (Elbow Method).
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_pca_2)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca_2[:, 0], y=X_pca_2[:, 1], hue=kmeans.labels_, palette='viridis', alpha=0.6)
plt.title("K Means Clustering (3 Clusters)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.show()


# Interpret the results of PCA by looking at the covariance matrix
h = .01
x_min, x_max = X_pca_2[:, 0].min() - 1, X_pca_2[:, 0].max() + 1
y_min, y_max = X_pca_2[:, 1].min() - 1, X_pca_2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(10, 6))
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(X_pca_2[:, 0], X_pca_2[:, 1], 'k.', markersize=2)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title("K Means Clustering (3 Clusters)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid()
plt.show()

for i in np.arange(2):
   print("Center of cluster", i+1, ":", centroids[i])

# Interpret the results of PCA by looking at the covariance matrix
cov_matrix_pca = pca_2.get_covariance()
plt.figure(figsize=(10, 6))
sns.heatmap(cov_matrix_pca, annot=True, cmap='coolwarm', fmt=".2f", mask=np.triu(np.ones_like(cov_matrix_pca, dtype=bool)), linewidths=0.5)
plt.title('Covariance Matrix (PCA)')
plt.show()

# Perform K Means Clustering on the 2 component PCA transformed data with clusters 
# ranging from 2 to 11 and plot the K Means inertia against the number of clusters (Elbow Method).
inertia = []
K_range = range(2, 12)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca_2)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Clusters')
plt.grid()
plt.show()

# Perform K Means Clustering on the 2 component PCA transformed data with the ideal number of clusters found
kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
kmeans.fit(X_pca_2)
y_kmeans = kmeans.predict(X_pca_2)

# Visualize the clusters on a scatter plot between 1st PCA and 2nd PCA component giving different colors to each cluster.
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca_2[:, 0], y=X_pca_2[:, 1], hue=y_kmeans, palette='viridis', alpha=0.6)
plt.title("K Means Clustering (4 Clusters)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.show()

