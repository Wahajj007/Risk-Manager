import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt


test_df = pd.read_csv('data/test_df_scored.csv')

print(test_df.shape)
print(test_df[['RobustMahalanobisDist', 'Class']].describe())
# Step 1: Select the top anomalous transactions
top_pct = 0.05
threshold = test_df['RobustMahalanobisDist'].quantile(1 - top_pct)
anomalous_df = test_df[test_df['RobustMahalanobisDist'] >= threshold].copy().reset_index(drop=True)

print(f"Selected {len(anomalous_df)} transactions above the {100*(1-top_pct):.0f}th percentile "
      f"(threshold distance: {threshold:.2f})")
print(anomalous_df['Class'].value_counts())

# Step 2: Build edges between transactions that are close in feature space
# Reuse the same features driving the anomaly score
graph_features = ['LogAmount', 'V1', 'V2', 'V3', 'V4', 'V14', 'V17']
X_anomalous = anomalous_df[graph_features].values

# Compute pairwise distances (manageable at ~2,868 points)
dist_matrix = euclidean_distances(X_anomalous)

# Connect pairs closer than some threshold - use a percentile of the distance
# distribution itself so this isn't an arbitrary guess
off_diagonal = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
edge_threshold = np.percentile(off_diagonal, 1)  # closest 1% of all pairs

print(f"Edge threshold (distance): {edge_threshold:.3f}")
print(f"This will create roughly {int(len(off_diagonal) * 0.01)} edges")

G = nx.Graph()
G.add_nodes_from(range(len(anomalous_df)))

n = len(anomalous_df)
for i in range(n):
    for j in range(i + 1, n):
        if dist_matrix[i, j] <= edge_threshold:
            G.add_edge(i, j)

print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Number of connected components: {nx.number_connected_components(G)}")

# Louvain community detection - needs python-louvain package
try:
    import community as community_louvain
except ImportError:
    print("python-louvain not installed. Run: pip install python-louvain")
    raise

partition = community_louvain.best_partition(G, random_state=42)
anomalous_df['CommunityID'] = anomalous_df.index.map(partition)

print(f"Number of communities detected: {anomalous_df['CommunityID'].nunique()}")

# For each community, compute size and fraud purity
community_stats = anomalous_df.groupby('CommunityID').agg(
    size=('Class', 'size'),
    fraud_count=('Class', 'sum')
)
community_stats['fraud_rate'] = community_stats['fraud_count'] / community_stats['size']

# Sort by fraud rate, show the most concentrated communities with meaningful size
community_stats_sorted = community_stats[community_stats['size'] >= 5].sort_values('fraud_rate', ascending=False)
print(community_stats_sorted.head(20))

# Use a layout that naturally separates connected clusters
pos = nx.spring_layout(G, seed=42, k=0.15, iterations=50)

fig, ax = plt.subplots(figsize=(14, 14))

# Color nodes by actual fraud label (ground truth) - this is your validation view
node_colors = ['red' if anomalous_df.loc[i, 'Class'] == 1 else 'lightblue' for i in G.nodes()]
node_sizes = [30 if anomalous_df.loc[i, 'Class'] == 1 else 15 for i in G.nodes()]

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax)
nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.5, ax=ax)

ax.set_title('Transaction Similarity Graph — Red = Confirmed Fraud, Blue = Normal\n'
             '(Top 5% most anomalous transactions, edges = high feature similarity)',
             fontsize=14)
ax.axis('off')

plt.tight_layout()
plt.savefig('fraud_ring_graph.png', dpi=150, bbox_inches='tight')
print("Saved fraud_ring_graph.png")
plt.close()