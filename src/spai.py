import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from helper import *

def preconditioner_with_sparcity_S(A: sp.csc_matrix, S: sp.csc_matrix, tol=1e-6) -> sp.csc_matrix:
    N = A.shape[0]
    
    # minimização
    data, rows, cols = [], [], []
    for k in range(N):
        indices = S[:,k].nonzero()[0] # linha dos elementos não nulos da coluna k
        if len(indices) == 0:
            continue

        Asub = A[indices, :][:, indices]
        Asub.tocsc()
        
        ek = np.zeros(N)
        ek[k] = 1
        eksub = ek[indices]
        
        mk = spla.lsqr(Asub, eksub)[0]

        for i, val in enumerate(mk):
            if abs(val) > tol:
                data.append(val)
                cols.append(k)
                rows.append(indices[i])
    
    return sp.csc_matrix((data, (rows, cols)), shape=A.shape)


def spaip(A: sp.csc_matrix, p: float, tol=1e-6) -> sp.csc_matrix:
    if not (0 <= p <= 2):
        raise ValueError("'p' deve estar entre 0 e 2")

    N = A.shape[0]
    A = A.copy()

    S = A.copy()
    S.data[:] = 1
    S.setdiag(0)
    S.eliminate_zeros()
    S1 = S.copy()

    if p == 0:
        # SPAI-0
        S_pattern = sp.eye(N, format="csc")

    elif p <= 1:
        # SPAI-p com 0 < p <= 1
        G = nx.from_scipy_sparse_array(S1)
        degrees = sorted(G.degree, key=lambda x: x[1])  # (node, degree)
        m = round((1 - p) * N)
        nodes_to_remove = [v for v, _ in degrees[:m]]

        S1_p = S1.tolil()
        for node in nodes_to_remove:
            S1_p[node, :] = 0
            S1_p[:, node] = 0
        S1_p = S1_p.tocsc()

        S_pattern = (sp.eye(N, format="csc") + S1_p)

    elif p <= 2:
        # SPAI-p com 1 < p <= 2
        S2 = S1 @ S1
        S2.setdiag(0)
        S2.eliminate_zeros()
        S2_star = S2.copy()

        # Subtrai S1 da S2
        diff = S2_star - S1
        diff.data = np.where(diff.data > 0, 1, 0)
        diff.eliminate_zeros()

        graus_S2 = np.array(diff.sum(axis=1)).flatten()
        ordem_S2 = np.argsort(graus_S2)

        m = round((2 - p) * N)
        nodes_to_remove = ordem_S2[:m]

        S2_p = S1.copy().tolil()
        for node in nodes_to_remove:
            S2_p[node, :] = 0
            S2_p[:, node] = 0
        S2_p = S2_p.tocsc()

        S_pattern = (sp.eye(N, format="csc") + S2_p)

    # retorn M tal que |A @ M - I| é minimo
    return preconditioner_with_sparcity_S(A, S_pattern, tol=tol)


if __name__ == "__main__":
    A = load_matrix("matrices/100.npz")
    p = 2
    S = spaip(A, p)

    show_sparcity_pattern(A @ A)
    show_sparcity_pattern(A)
    show_sparcity_pattern(S)
    
