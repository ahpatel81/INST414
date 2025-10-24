import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

edges = [("funny", "AskReddit"), ("AskReddit", "todayilearned"),
         ("gaming", "funny"), ("worldnews", "news"),
         ("science", "todayilearned"), ("movies", "funny"),
         ("memes", "funny"), ("gaming", "memes"),
         ("aww", "funny"), ("Music", "funny"),
         ("worldnews", "AskReddit"), ("science", "worldnews")]

G = nx.DiGraph()
G.add_edges_from(edges)

pageranks = nx.pagerank(G)
pr = pd.Series(pageranks, name="pagerank")

top = pr.sort_values(ascending=False).reset_index()
top.columns = ["subreddit", "pagerank"]

indegree_map = dict(G.in_degree())
indegree_series = pd.Series(indegree_map, name="in_degree")

df = top.merge(indegree_series.reset_index().rename(columns={"index": "subreddit"}), on="subreddit")
df = df[["subreddit", "pagerank", "in_degree"]]

print("\n Top Subreddits by PageRank")
print(df.head(3).to_string(index=False))

plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42)

sizes = [5000 * pageranks.get(n, 0.01) for n in G.nodes()]
nx.draw(G, pos=pos, with_labels=True, node_size=sizes, arrows=True, arrowsize=10)

plt.title("Reddit Subreddit Network")
plt.tight_layout()
plt.savefig("network.png", dpi=200)
plt.show()
