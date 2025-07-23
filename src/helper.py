from matplotlib import pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse import load_npz
import networkx as nx
import random
from scipy.sparse import csc_matrix
import numpy as np

def save_sparcity_figure(A, path: str) -> None:
    plt.spy(A)
    plt.axis("off")
    plt.savefig(path)


def get_sparcity_pattern(A: sp.csc_matrix):
    S = (A != 0).astype(int)
    return sp.csc_matrix(S)


def show_sparcity_pattern(A: sp.csc_matrix):
    plt.spy(A.toarray())
    plt.show()


def identity_difference(A: sp.csc_matrix) -> float:
    N = A.shape[0]
    return spla.norm(A - sp.eye(N))


# def condition_number(A: sp.csc_matrix) -> float:
#     A_csc = A.tocsc()
#     Ainv = spla.inv(A_csc)
#     return spla.norm(A_csc) * spla.norm(Ainv)


def condition_number(A: sp.csc_matrix) -> float:
    # Estima norma 1 de A
    Aop = spla.LinearOperator(shape=A.shape, matvec=A.dot, rmatvec=lambda x: A.T @ x)
    norm_A = spla.onenormest(Aop)

    # Fatoração LU de A (reutilizável e rápida)
    lu = spla.splu(A)

    # Define operador para A^{-1} usando LU
    def Ainv(x): return lu.solve(x)
    def Ainv_T(x): return lu.solve(x, 'T')

    Ainv_op = spla.LinearOperator(shape=A.shape, matvec=Ainv, rmatvec=Ainv_T)
    norm_Ainv = spla.onenormest(Ainv_op)

    return norm_A * norm_Ainv


def benchmark_identity_diference(A: sp.csc_matrix, M: sp.csc_matrix):
    N = A.shape[0]
    print("=== Identity difference benchmark ===")
    print(f"|A - I| = {spla.norm(A - sp.eye(N))}")
    print(f"|M - I| = {spla.norm(M - sp.eye(N))}")
    print("\n")


def benchmark_condition_number(A: sp.csc_matrix, M: sp.csc_matrix):
    N = A.shape[0]
    A_csc = A.tocsc()
    Ainv = spla.inv(A_csc)
    cond_A = spla.norm(A_csc) * spla.norm(Ainv)

    M_csc = M.tocsc()
    Minv = spla.inv(M_csc)
    cond_M = spla.norm(M_csc) * spla.norm(Minv)

    print("=== Condition number benchmark ===")
    print(f"cond(A) = {cond_A}")
    print(f"cond(M) = {cond_M}")
    print("\n")


def load_matrix(filepath: str):
    M = load_npz(filepath)
    return M


def build_random_matrix(N, m):
    G = nx.barabasi_albert_graph(N, m)

    M = nx.to_numpy_array(G)

    for i in range (0,N):
        k = 0
        for j in range (0,N):
            if M[i,j] != 0: 
                k += 1
                M[i,j] += 2*random.random() 
        M[i,i] = k  
    
    return csc_matrix(M)