import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from scipy.sparse import csr_matrix, eye, issparse
from helper import timeit


def make_spai_pattern(A: np.ndarray, p: float) -> np.ndarray:
    """
    Computa o padrão de espacidade de SPAI-p para A e 0 <= p <= 2
    """
    if not (0 <= p <= 2):
        raise ValueError("p must be between 0 and 2")

    n = A.shape[0]
    A = csr_matrix(A) if not issparse(A) else A

    S = (A != 0).astype(int).tocsr()
    I = eye(n, dtype=int, format='csr')
    S1 = (S - I).tolil()  # Use LIL for row/col deletion

    if p <= 1:
        m = round((1 - p) * n)
        deg = np.array(S1.sum(axis=1)).flatten()
        sort_deg = np.argsort(deg)
        to_remove = sort_deg[:m]

        for idx in to_remove:
            S1[idx, :] = 0
            S1[:, idx] = 0

        return (I + S1.tocsr()).toarray()
    
    else:
        m = round((2 - p) * n)
        S1_csr = S1.tocsr()
        S2 = (S1_csr @ S1_csr).tolil()
        S2.setdiag(0)

        deg2 = np.array(S2.sum(axis=1)).flatten()
        sort_deg2 = np.argsort(deg2)
        to_remove = sort_deg2[:m]

        for idx in to_remove:
            S2[idx, :] = 0
            S2[:, idx] = 0

        M = I + S1_csr + S2.tocsr()
        return (M > 0).astype(int).toarray()
    

@timeit
def fast_spai(A: sp.csc_matrix, p: float, tol: float = 1e-6) -> sp.csc_matrix:
    """
    Retorna o precondicionador a esquerda da matriz A com esparcidade dada por SPAI-p
    - M é uma aproximação de A^{-1} a esquerda 
    """
    n = A.shape[0]
    A = A.tocsc().transpose()
    P = make_spai_pattern(A.toarray(), p)
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
        result = lsqr(A_sub, e_j, atol=tol, btol=tol)
        x = result[0]
        M_cols.extend(x)
        row_indices.extend(pattern)
        col_ptrs.append(len(M_cols))

    M = sp.csc_matrix((M_cols, row_indices, col_ptrs), shape=(n, n)).transpose()
    return M


@timeit
def make_spai_0(A: sp.csc_matrix):
    """
    Computa o precondicionador de M SPAI-0 de A
    """
    N = A.shape[0]
    A = A.toarray()
    D = np.zeros(shape=A.shape)

    for i in range(N):
        D[i,i] = A[i,i] / np.sum(A[i,:] * A[i,:])
    
    return sp.csc_matrix(D)
    