import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset
df = pd.read_csv("emails.csv")
print("Dataset shape:", df.shape)
display(df.head())

# 2. Quick check for missing values
print("Missing values per column:\n", df.isnull().sum())

# 3. Define feature matrix X and label vector y
#    Adjust indexing if your dataset columns differ.
#    Here we assume columns 1..3000 (word frequencies) are features and last column is label.
if df.shape[1] < 2:
    raise ValueError("Dataset has too few columns. Check the CSV file.")

X = df.iloc[:, 1:3001]   # change if actual number of feature columns is different
y = df.iloc[:, -1].values

print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)

# 4. Numeric columns (used for outlier analysis if needed)
df_numeric = df.select_dtypes(include=['int64', 'float64']).copy()
print("Numeric columns count:", len(df_numeric.columns))

# 5. Optional: visualize top outlier features (IQR method)
Q1 = df_numeric.quantile(0.25)
Q3 = df_numeric.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outlier_mask = ((df_numeric < lower) | (df_numeric > upper))
outlier_counts = outlier_mask.sum().sort_values(ascending=False)

topN = 12
top_features = outlier_counts.head(topN).index.tolist()
if len(top_features) > 0:
    plt.figure(figsize=(16,6))
    sns.boxplot(data=df_numeric[top_features])
    plt.title(f"Boxplots for top {topN} features by outlier count")
    plt.xticks(rotation=45, ha='right')
    plt.show()

# 6. Split data (do this BEFORE scaling to avoid data leakage)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 7. Scale features (important for SVM and KNN)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 8. Train & evaluate SVM (on scaled data)
svc = SVC(C=1.0, kernel='rbf', gamma='auto')
svc.fit(X_train_s, y_train)
svc_pred = svc.predict(X_test_s)
print("SVM Accuracy:", accuracy_score(y_test, svc_pred))
print("SVM Classification Report:\n", classification_report(y_test, svc_pred, zero_division=0))
print("SVM Confusion Matrix:\n", confusion_matrix(y_test, svc_pred))

# 9. Train & evaluate KNN (on scaled data) — one example k
knn = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
knn.fit(X_train_s, y_train)
knn_pred = knn.predict(X_test_s)
print("KNN Accuracy:", accuracy_score(y_test, knn_pred))
print("KNN Classification Report:\n", classification_report(y_test, knn_pred, zero_division=0))
print("KNN Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))

# 10. Try multiple k values and compare
ks = [1, 3, 5, 7, 9]
results = {}
for k in ks:
    knn_k = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn_k.fit(X_train_s, y_train)
    y_pred_k = knn_k.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred_k)
    results[k] = acc
    print(f"K = {k} → Accuracy = {acc:.4f}")

print("All results:", results)
