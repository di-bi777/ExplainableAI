import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Kaggleの「Mall Customers」データセットを読み込む
file_path = 'C:\\Users\\FujimotoD\\OneDrive\\documents\\intern\\ExplanableAI\\Mall_Customers.csv'
data = pd.read_csv(file_path)

# 必要な列を選択する
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# KMeansクラスタリングを適用する
kmeans = KMeans(n_clusters=5, random_state=0)
clusters = kmeans.fit_predict(X)

# クラスタリング結果をデータフレームに追加する
data['Cluster'] = clusters

# クラスタリング結果を色分けして表示する
plt.figure(figsize=(10, 6))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Mall Customers Clustering')
plt.colorbar(label='Cluster')
plt.show()
