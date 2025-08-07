import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from helper import build_random_matrix, make_spai_pattern, plot_spai_interpolation
from scipy.sparse.linalg import lsqr


def plot_graph_adjacency():
    sns.set_theme(style="white")

    N = 10
    m = 4
    G = nx.barabasi_albert_graph(N, m)
    A = nx.adjacency_matrix(G)

    fig, axs = plt.subplots(nrows=1, ncols=2)
    fig.suptitle("Visualização gráfo de Barabási–Albert", fontsize=16)

    # Draw the graph
    nx.draw(
        G,
        ax=axs[0],
        node_color="black",
        node_size=300,
        edge_color="gray",
        with_labels=True,
        font_size=10,
        font_color="white"
    )
    axs[0].set_title("Estrutura em Grafo (N=10, m=4)", fontsize=12)
    axs[0].axis("off")

    # Show the adjacency matrix
    axs[1].spy(A.toarray())
    axs[1].set_title("Matriz de Adjacência", fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()


def plot_degree_histogram():
    N = 1000
    m = 4
    G = nx.barabasi_albert_graph(N, m)

    degrees = [d for _, d in G.degree()]

    plt.hist(degrees, bins=max(degrees), color='black', edgecolor='white', alpha=0.8)

    plt.xlabel("Grau k")
    plt.xlim((0, 40))
    plt.ylabel("Frequência")
    plt.title(f"Histograma da distribuição de grau N={N}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_paralel_computing():
    N, m = 10, 4
    A = build_random_matrix(N, m).toarray()
    
    fig, axs = plt.subplots(nrows=3, ncols=5)
    fig.suptitle("Computação paralela por colunas")

    for i in range(2):
        for j in range(5):
            index = i * 5 + j
            S = np.zeros(shape=A.shape)
            S[:,index] = A[:,index]
            axs[i][j].spy(S)
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
            axs[i][j].set_title(f"coluna {index}")
    
    for j in range(5):
        if j != 2:
            axs[2][j].axis("off")
        else:
            axs[2][2].spy(A)
            axs[2][2].set_xticks([])
            axs[2][2].set_yticks([])
            axs[2][2].set_title(f"M^t")
    
    plt.show()
        

def plot_minimization_computing():
    N, m = 10, 4
    A = build_random_matrix(N, m)
    P = make_spai_pattern(A, 0.5)

    M_cols = []
    row_indices = []
    col_ptrs = [0]


    j = 0
    pattern = np.where(P[:,j])[0]
    A_sub = A[:, pattern].toarray()
    e_j = np.zeros(N)
    e_j[j] = 1
    result = lsqr(A_sub, e_j)
    x = result[0]

    M_cols.extend(x)
    row_indices.extend(pattern)
    col_ptrs.append(len(M_cols))
    print(row_indices)
    row_indices.extend(pattern)
    print(row_indices)


def plot_A_spai1_spai2():
    N, m = 10, 4
    A = build_random_matrix(N, m)
    P1 = make_spai_pattern(A, 1)
    P2 = make_spai_pattern(A, 2)

    fig, axs = plt.subplots(nrows=1, ncols=3)
    fig.suptitle("Sparcidade de A, SPAI-1 e SPAI-2")

    axs[0].spy(A.toarray())
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_title("A")

    axs[1].spy(P1)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_title("SPAI-1")

    axs[2].spy(P2)
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    axs[2].set_title("SPAI-2")

    plt.show()
    

def plot_computacap_paralela():
    # Criar uma matriz esparsa 10x10
    M = build_random_matrix(N=10, m=4).toarray()
    M = M.T

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 6, width_ratios=[3, 1, 1, 1, 1, 1])

    # --- Gráfico da esquerda: esparsidade completa ---
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.spy(M)
    ax0.set_title("Esparsidade de M^t")
    ax0.set_xlabel("Colunas")
    ax0.set_ylabel("Linhas")

    # --- Gráficos da direita: esparsidade de cada linha ---
    for i in range(10):
        row = i // 5  # linha do grid (0 ou 1)
        col = i % 5   # coluna do grid (0 a 4)

        # Criar uma cópia com apenas a i-ésima linha
        M_i = np.zeros_like(M)
        M_i[i, :] = M[i, :]

        ax = fig.add_subplot(gs[row, col + 1])
        ax.spy(M_i)
        ax.set_title(f"Linha {i}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def plot_matrix_with_values():
    # Criar uma matriz esparsa 10x10 de exemplo
    A = build_random_matrix(N=8, m=3)
    P = make_spai_pattern(A, p=1)
    P = P - np.eye(N=P.shape[0])  # remover diagonal

    # Calcula P^2
    P2 = P @ P

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    fig.suptitle("Matrizes e seus grafos", fontsize=16)

    # --- Plot da matriz P ---
    axs[0, 0].spy(P)
    axs[0, 0].set_title("S1")
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    axs[0, 0].grid(False)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i, j] != 0:
                axs[0, 0].text(j, i, f"{P[i, j]:.0f}", ha='center', va='center', fontsize=8, color='white')

    # --- Plot da matriz P² ---
    axs[0, 1].spy(P2)
    axs[0, 1].set_title("S2 = S1²")
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    axs[0, 1].grid(False)
    for i in range(P2.shape[0]):
        for j in range(P2.shape[1]):
            if P2[i, j] != 0:
                axs[0, 1].text(j, i, f"{P2[i, j]:.0f}", ha='center', va='center', fontsize=8, color='white')

    # --- Grafo de S1 ---
    G1 = nx.from_numpy_array(P, create_using=nx.DiGraph)
    pos = nx.spring_layout(G1, seed=42)
    nx.draw(G1, pos, ax=axs[1, 0], with_labels=True, node_color='lightblue', arrows=True, node_size=500)

    # --- Grafo de S2 ---
    G2 = nx.from_numpy_array(P2, create_using=nx.DiGraph)
    nx.draw(G2, pos, ax=axs[1, 1], with_labels=True, node_color='lightgreen', arrows=True, node_size=500)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_interpolation():
    pass


if __name__ == "__main__":
    from spai import fast_spai
    A = build_random_matrix(15, 4)
    print(np.allclose(A.toarray(), A.T.toarray(), rtol=1e-6))


