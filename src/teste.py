import numpy as np
from helper import plot_D_A_A2, plot_spai_interpolation
from helper import build_random_matrix
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, eye, issparse



def make_spai_pattern(A: np.ndarray, p: float) -> np.ndarray:
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



if __name__ == "__main__":
   A = build_random_matrix(15, 4).toarray()
   plot_D_A_A2(A, save=True)
   plot_spai_interpolation(A, save=True)