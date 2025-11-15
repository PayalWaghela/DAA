import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load data
df = pd.read_csv('sales_data_sample.csv', encoding='unicode_escape')
print(df.shape)
df.head()

# 2. Drop unused columns
to_drop = ['ADDRESSLINE1', 'ADDRESSLINE2', 'STATE', 'POSTALCODE', 'PHONE']
df = df.drop(columns=[c for c in to_drop if c in df.columns])

# 3. Select numeric columns only
df_numeric = df.select_dtypes(include=['int64', 'float64']).copy()
print("Numeric columns:", df_numeric.columns.tolist())

# 4. Visualize boxplots (optional)
plt.figure(figsize=(12,6))
sns.boxplot(data=df_numeric)
plt.title("Outlier Detection using Boxplots (Numeric Columns Only)")
plt.show()

# 5. Detect and cap outliers using IQR (create df_capped)
Q1 = df_numeric.quantile(0.25)
Q3 = df_numeric.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Clip each column to [lower, upper]
df_capped = df_numeric.clip(lower=lower, upper=upper, axis=1)
print("Applied clipping to numeric features to handle outliers.")

# 6. Normalize (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_capped)
df_normalized = pd.DataFrame(X_scaled, columns=df_capped.columns)
display(df_normalized.head())

# 7. Elbow method (inertia) to choose K
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K, inertia, marker='o')
plt.title("Elbow Method to Determine Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

# 8. Silhouette scores for K = 2..10
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    print(f"K = {k} → Silhouette Score = {sil:.4f}")

# 9. Visualize clusters for chosen K (example K=3)
chosen_k = 3
kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# Make sure the columns SALES and MSRP exist before plotting
if 'SALES' in df_capped.columns and 'MSRP' in df_capped.columns:
    plt.figure(figsize=(8,6))
    sns.scatterplot(
        x=df_capped['SALES'],
        y=df_capped['MSRP'],
        hue=labels,
        palette='Set2',
        legend='full'
    )
    plt.title(f"K-Means Clustering Visualization (K = {chosen_k})")
    plt.xlabel("SALES")
    plt.ylabel("MSRP")
    plt.legend(title='Cluster')
    plt.show()
else:
    print("SALES or MSRP column not found in numeric columns; cannot plot SALES vs MSRP.")
