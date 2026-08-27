import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import plotly.graph_objects as go



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

print("\n=== SENSITIVITY CHECK: different top-% and edge thresholds ===")

def run_sensitivity_check(top_pct, edge_percentile):
    threshold_val = test_df['RobustMahalanobisDist'].quantile(1 - top_pct)
    subset = test_df[test_df['RobustMahalanobisDist'] >= threshold_val].copy().reset_index(drop=True)

    X_sub = subset[graph_features].values
    dist_mat = euclidean_distances(X_sub)
    off_diag = dist_mat[np.triu_indices_from(dist_mat, k=1)]
    edge_thresh = np.percentile(off_diag, edge_percentile)

    G_sub = nx.Graph()
    G_sub.add_nodes_from(range(len(subset)))
    n = len(subset)
    for i in range(n):
        for j in range(i + 1, n):
            if dist_mat[i, j] <= edge_thresh:
                G_sub.add_edge(i, j)

    partition_sub = community_louvain.best_partition(G_sub, random_state=42)
    subset['CommunityID'] = subset.index.map(partition_sub)

    stats = subset.groupby('CommunityID').agg(size=('Class', 'size'), fraud_count=('Class', 'sum'))
    stats['fraud_rate'] = stats['fraud_count'] / stats['size']
    pure_communities = stats[(stats['size'] >= 5) & (stats['fraud_rate'] >= 0.9)]

    print(f"top_pct={top_pct}, edge_percentile={edge_percentile}: "
          f"{len(subset)} nodes, {len(pure_communities)} communities with >=90% fraud purity "
          f"(covering {pure_communities['fraud_count'].sum()} fraud transactions)")

# Test a few different parameter combinations
run_sensitivity_check(0.05, 1)   # your original settings
run_sensitivity_check(0.03, 1)   # tighter anomaly filter
run_sensitivity_check(0.07, 1)   # looser anomaly filter
run_sensitivity_check(0.05, 2)   # looser edge threshold


# Use the original graph (G, anomalous_df) with the spring layout positions already computed
edge_x, edge_y = [], []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.3, color='lightgray'),
    hoverinfo='none',
    mode='lines'
)

node_x = [pos[i][0] for i in G.nodes()]
node_y = [pos[i][1] for i in G.nodes()]
node_class = [anomalous_df.loc[i, 'Class'] for i in G.nodes()]
node_amount = [anomalous_df.loc[i, 'Amount'] for i in G.nodes()]
node_dist = [anomalous_df.loc[i, 'RobustMahalanobisDist'] for i in G.nodes()]
node_community = [partition[i] for i in G.nodes()]

hover_text = [
    f"Class: {'FRAUD' if c == 1 else 'Normal'}<br>Amount: ${a:.2f}<br>"
    f"Anomaly dist: {d:.2f}<br>Community: {comm}"
    for c, a, d, comm in zip(node_class, node_amount, node_dist, node_community)
]

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    text=hover_text,
    marker=dict(
        size=[8 if c == 1 else 4 for c in node_class],
        color=['red' if c == 1 else 'lightblue' for c in node_class],
        line=dict(width=0.5, color='white')
    )
)

fig = go.Figure(data=[edge_trace, node_trace])
fig.update_layout(
    title='Transaction Similarity Graph (Interactive) — Hover for details',
    showlegend=False,
    hovermode='closest',
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    height=800
)

fig.write_html('fraud_ring_graph_interactive.html')
print("Saved interactive graph to fraud_ring_graph_interactive.html")