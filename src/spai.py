import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from helper import *

def compute_spai(A: sp.csr_matrix, p: float, tol=1e-6) -> sp.csr_matrix:
    if not sp.isspmatrix_csr(A):
        raise ValueError(f"A deve ter classe {sp.csr_matrix}")

    N = A.shape[0]
    A = A.copy()

    # removendo a fracao 'p' dos vertices
    if p > 0:
        S = A.copy()
        S.data[:] = 1
        S1 = S - sp.eye(N, format='csr')
        
        graus = np.array(S1.sum(axis=1)).flatten()
        ordem = np.argsort(graus)
        m = int(np.floor(p * N))
        vertices_removidos = ordem[:m]

        # Convert to LIL for efficient row/column modifications
        A_lil = A.tolil()
        for v in vertices_removidos:
            A_lil[v, :] = 0
            A_lil[:, v] = 0
        A = (sp.eye(N, format='csr') + A_lil).tocsr()

    # minimizacao
    M_data = []
    M_rows = []
    M_cols = []

    for i in range(N):
        # montando vetor 'ei' da miniminação de |Ami - ei|
        e_i = np.zeros(N)
        e_i[i] = 1

        row_inds = A[:, i].nonzero()[0]
        if len(row_inds) == 0:
            continue

        A_sub = A[row_inds, :][:, row_inds].toarray()
        e_sub = e_i[row_inds]

        # minimos quadrados
        m_i = spla.lsqr(A_sub, e_sub)[0]

        # retornando M ao tamanho de A
        for local_idx, val in enumerate(m_i):
            if abs(val) > tol:
                global_row = row_inds[local_idx]
                M_rows.append(global_row)
                M_cols.append(i)
                M_data.append(val)

    M = sp.csr_matrix((M_data, (M_rows, M_cols)), shape=A.shape)
    return M


if __name__ == "__main__":
    from build_random_matrix import build_random_matrix
    A = build_random_matrix(10, 4)
    S = sp.csr_matrix(get_sparcity_pattern(A, 0))

    M1 = compute_spai(A, 1)
    M2 = compute_spai(A, 0)

    show_sparcity_pattern([M1, M2, S])
    
