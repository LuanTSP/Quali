import numpy as np
import networkx as nx
from math import floor
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr


def compute_spaip_pattern(A: np.ndarray, p: float) -> np.ndarray:
    if not (0 <= p <= 2):
        raise ValueError("p must be between 0 and 2")

    n = A.shape[0]

    G1 = nx.Graph()
    G1.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and A[i, j] != 0:
                G1.add_edge(i, j)

    if p <= 1:
        m = round((1 - p) * n)
        deg = dict(G1.degree())
        to_remove = sorted(deg, key=deg.get)[:m]
        G1.remove_nodes_from(to_remove)

        P = np.eye(n, dtype=int)
        for i, j in G1.edges():
            P[i, j] = 1
            P[j, i] = 1
        return P

    else:
        G2 = nx.Graph()
        G2.add_nodes_from(G1.nodes())

        for node in G1.nodes():
            neighbors = list(G1.neighbors(node))
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    u, v = neighbors[i], neighbors[j]
                    if u != v:
                        G2.add_edge(u, v)

        for node in G1.nodes():
            if G1.degree[node] > 0:
                G2.add_edge(node, node)

        m = floor((2 - p) * n)
        deg2 = dict(G2.degree())
        to_remove = sorted(deg2, key=deg2.get)[:m]
        G2.remove_nodes_from(to_remove)

        P = np.eye(n, dtype=int)
        for i, j in G2.edges():
            P[i, j] = 1
            P[j, i] = 1
        return P


def fast_spai(A: sp.csc_matrix, p: float, tol: float = 1e-6) -> sp.csc_matrix:
    n = A.shape[0]
    A = A.tocsc()
    P = compute_spaip_pattern(A.toarray(), p)
    M_cols = []
    row_indices = []
    col_ptrs = [0]

    for j in range(n):
        pattern = np.where(P[:, j])[0]
        if len(pattern) == 0:
            col_ptrs.append(col_ptrs[-1])
            continue
        A_sub = A[:, pattern]
        e_j = np.zeros(n)
        e_j[j] = 1
        result = lsqr(A_sub, e_j, atol=tol, btol=tol, iter_lim=1000)
        x = result[0]
        M_cols.extend(x)
        row_indices.extend(pattern)
        col_ptrs.append(len(M_cols))

    M = sp.csc_matrix((M_cols, row_indices, col_ptrs), shape=(n, n))
    return M