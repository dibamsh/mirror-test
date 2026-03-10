import numpy as np
import networkx as nx
from collections import Counter, defaultdict


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


def weighted_pagerank(G, alpha=0.85, max_iter=1000, tol=1e-9):
    """PageRank where each edge weight = 1 / (relation frequency).
    Rare relations contribute more; common relations contribute less."""
    G = G.copy()

    rel_counts = Counter()
    for u, v, data in G.edges(data=True):
        rel_counts[data['relation_name']] += 1
    total = G.number_of_edges()
    for u, v, key, data in G.edges(keys=True, data=True):
        freq = rel_counts[data['relation_name']] / total
        G[u][v][key]['weight'] = 1.0 / freq

    return _weighted_power_iter(G, alpha, max_iter, tol)


def domainrange_pagerank(G, alpha=0.85, max_iter=1000, tol=1e-9):
    """PageRank where each edge weight is the average structural similarity
    (domain-domain + range-range overlap, dd+rr) of its relation to all others."""
    G = G.copy()

    triples = [(u, data['relation_name'], v)
               for u, v, data in G.edges(data=True)]

    D = defaultdict(set)
    R = defaultdict(set)
    for s, p, o in triples:
        D[p].add(s)
        R[p].add(o)

    predicates = sorted(set(p for _, p, _ in triples))
    pred_idx = {p: i for i, p in enumerate(predicates)}
    n = len(predicates)

    structural = np.zeros((n, n))
    for i, pi in enumerate(predicates):
        for j, pj in enumerate(predicates):
            if i != j:
                dd = len(D[pi] & D[pj])
                rr = len(R[pi] & R[pj])
                max_possible = max(len(D[pi]), len(D[pj]), len(R[pi]), len(R[pj]), 1)
                structural[i][j] = (dd + rr) / max_possible

    for u, v, key, data in G.edges(keys=True, data=True):
        rel = data['relation_name']
        if rel in pred_idx:
            idx = pred_idx[rel]
            avg = np.mean(structural[idx])
            G[u][v][key]['weight'] = max(avg, 1e-10)

    return _weighted_power_iter(G, alpha, max_iter, tol)


def _weighted_power_iter(G, alpha, max_iter, tol):
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {}, 0

    node_idx = {node: i for i, node in enumerate(nodes)}
    x = np.ones(n) / n

    total_weight = sum(G[u][v][key].get('weight', 1.0)
                       for u, v, key in G.edges(keys=True))
    scale = n / max(total_weight, 1.0)

    out_deg = {}
    for node in nodes:
        out_deg[node] = sum(
            G[node][nb][key].get('weight', 1.0) * scale
            for nb in G.successors(node)
            for key in G[node][nb]
        )

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
                        w = G[node][neighbor][key].get('weight', 1.0) * scale
                        x[j] += alpha * x_last[i] * w / out_deg[node]
        x = x / x.sum()
        if np.abs(x - x_last).sum() < n * tol:
            return {nodes[i]: x[i] for i in range(n)}, iteration + 1

    return {nodes[i]: x[i] for i in range(n)}, max_iter
