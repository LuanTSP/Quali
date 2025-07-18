import numpy as np
import networkx as nx
from math import floor


def build_spai_p01_matrix(A, p):
    N = A.shape[0]

    G = nx.from_numpy_array(A)

    degree_list = sorted(G.degree, key=lambda x: x[1])
    V = [v for v, _ in degree_list]

    m = floor(p * N)
    nodes_to_remove = V[:m]
    nodes_remaining = sorted(set(range(N)) - set(nodes_to_remove))

    G_sp = G.copy()
    G_sp.remove_nodes_from(nodes_to_remove)

    A_p_reduced = nx.to_numpy_array(G_sp, nodelist=nodes_remaining)

    A_p = np.zeros((N, N), dtype=int)
    for i_new, i_old in enumerate(nodes_remaining):
        for j_new, j_old in enumerate(nodes_remaining):
            A_p[i_old, j_old] = A_p_reduced[i_new, j_new]
    

    return A_p


if __name__ == "__main__":
    # Teste para uma matrix A
    N = 100
    m = 4
    p = 0.2

    G_original = nx.barabasi_albert_graph(N, m)
    A = nx.to_numpy_array(G_original)

    # Construção do SPAI-p
    A_spai_p = build_spai_p01_matrix(A, p)

    # Informações matrix SPAI-p
    rank = np.linalg.matrix_rank(A_spai_p)
    print(f"Rank da matriz A: {np.linalg.matrix_rank(A):.2f}")
    print(f"Rank da matriz SPAI-p com p = {p}: {rank:.2f}")
    print(f"Dimensão da A: {A.shape}")
    print(f"Dimensão da matriz SPAI-p com p = {p}: {A_spai_p.shape}")
    print(f"Número de condição de SPAI-p com p = {p}: {np.linalg.cond(A_spai_p):.2f}")
    print(f"Número de condição de A: {np.linalg.cond(A):.2f}")
