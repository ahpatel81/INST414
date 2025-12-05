import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


np.random.seed(42)

num_players = 100

fortnite_data = {
    "username": [f"Player{i+1}" for i in range(num_players)],
    
    "matches": np.random.randint(10, 5000, num_players),
    
    "wins": np.random.randint(0, 1200, num_players),
    
    "kills": np.random.randint(0, 8000, num_players),

    "kd": np.round(np.random.uniform(0.5, 6.0, num_players), 2),

    "winrate": np.round(np.random.uniform(0.5, 30.0, num_players), 2),
    
    "score": np.random.randint(100, 20000, num_players),
}

df = pd.DataFrame(fortnite_data)

df["deaths"] = np.round(df["kills"] / df["kd"], 2)



features = ["matches", "wins", "kills", "deaths", "kd", "winrate", "score"]
X = df[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Finding the value of k - assisted by ChatGPT

inertias = []
silhouette_scores = []
K = range(2, 10) 

for k in K:
    model = KMeans(n_clusters = k, random_state = 42, n_init = 10)
    labels = model.fit_predict(X_scaled)
    
    inertias.append(model.inertia_)


plt.figure(figsize=(6, 4))
plt.plot(K, inertias, marker="o")
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.show()

print("\nK values tested:", list(K))
print("Inertias:", inertias)

# Back to analysis after finding k

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

df["cluster"] = labels

print("\nCluster Assignments:")
print(df[["username", "cluster"]])


cluster_summary = df.groupby("cluster")[features].mean()
print("\nCluster Stats:")
print(cluster_summary)