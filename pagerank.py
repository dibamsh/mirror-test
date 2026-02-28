import numpy as np
import networkx as nx


def standard_pagerank(G, alpha=0.85, max_iter=1000, tol=1e-9):
    G = G.to_directed()
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {}, 0

    node_idx = {node: i for i, node in enumerate(nodes)}
    x = np.ones(n) / n
    out_deg = {node: G.out_degree(node) for node in nodes}
    dangling = [node_idx[nd] for nd in nodes if out_deg[nd] == 0]

    for iteration in range(max_iter):
        x_last = x.copy()
        x = np.zeros(n)
        dangling_sum = sum(x_last[i] for i in dangling)
        for i in range(n):
            x[i] = (1 - alpha) / n + alpha * dangling_sum / n
        for node in nodes:
            i = node_idx[node]
            if out_deg[node] > 0:
                for neighbor in G.successors(node):
                    j = node_idx[neighbor]
                    for key in G[node][neighbor]:
                        x[j] += alpha * x_last[i] / out_deg[node]
        x = x / x.sum()
        if np.abs(x - x_last).sum() < n * tol:
            return {nodes[i]: x[i] for i in range(n)}, iteration + 1

    return {nodes[i]: x[i] for i in range(n)}, max_iter
