from src.helper import *
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np
import networkx as nx
from math import floor, ceil

def preconditioner_with_sparcity_S(A: sp.csc_matrix, S: sp.csc_matrix, tol=1e-6) -> sp.csc_matrix:
    N = A.shape[0]
    # minimização
    data = []
    rows = []
    cols = []
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


def spaip(A: sp.csc_matrix, p: float, tol=1e-6):
    N = A.shape[0]
    S = A.copy()
    S.data[:] = 1
    S = S - sp.eye(N)
    
    G = nx.from_scipy_sparse_array(S)
    degrees = sorted(G.degree, key=lambda x: x[1])
    m = round((1 - p) * N)
    
    S = S.tolil()
    for i in range(m):
        line = degrees[i][0]
        S[:,line] = 0
        S[line,:] = 0
    S.tocsr()
    S = sp.eye(N) + S

    return preconditioner_with_sparcity_S(A, S, tol)


if __name__ == "__main__":
    A = load_matrix("matrices/10.npz")
    M1 = spaip(A, 1)
    M09 = spaip(A, 0.9)
    M08 = spaip(A, 0.8)
    M07 = spaip(A, 0.7)
    M0 = spaip(A, 0)
    show_sparcity_pattern([M1, M09, M08, M07, M0], ["1", "0.9", "0.8", "0.7", "0"])
