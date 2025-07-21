from matplotlib import pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np

def save_sparcity_figure(A, path: str) -> None:
    plt.spy(A)
    plt.axis("off")
    plt.savefig(path)


def get_sparcity_pattern(A: sp.csr_matrix, p=0):
    N = A.shape[0]
    S = (A != 0).astype(int)
    S1 = S - np.eye(N, dtype=int)
    graus = np.sum(S1, axis=1).flatten()
    ordem = np.argsort(graus)
    m = int(np.floor(p * N))

    vertices_removidos = ordem[:,:m]

    S1p = np.copy(S1)
    for v in vertices_removidos:
        S1p[v, :] = 0
        S1p[:, v] = 0

    return np.eye(N, dtype=int) + S1p


def show_sparcity_pattern(A, titles=None, p=0):
    if type(A) == list:
        L = len(A)
        _, axs = plt.subplots(ncols=L, nrows=1)

        for i in range(L):
            axs[i].spy(get_sparcity_pattern(A[i], p))
            axs[i].set_title(titles[i])
    
    else:
        plt.spy(get_sparcity_pattern(A, p))
        plt.title(titles)
    
    plt.show()


def identity_difference(A: sp.csr_matrix) -> float:
    N = A.shape[0]
    return spla.norm(A - sp.eye(N))


def condition_number(A: sp.csr_matrix) -> float:
    N = A.shape[0]
    A_csc = A.tocsc()
    Ainv = spla.inv(A_csc)
    return spla.norm(A_csc) * spla.norm(Ainv)
    

def benchmark_identity_diference(A: sp.csr_matrix, M: sp.csr_matrix):
    N = A.shape[0]
    print("=== Identity difference benchmark ===")
    print(f"|A - I| = {spla.norm(A - sp.eye(N))}")
    print(f"|M - I| = {spla.norm(M - sp.eye(N))}")
    print("\n")


def benchmark_condition_number(A: sp.csr_matrix, M: sp.csr_matrix):
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