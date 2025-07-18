import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def build_spai(A, tol=1e-5):
    """
    Constroi o preconsicionador SPAI para a matriz A tal que A * M \approx I
    """
    N = A.shape[0]
    M_data = []
    M_rows = []
    M_cols = []

    for i in range(N):
        e_i = np.zeros(N)
        e_i[i] = 1

        row_inds = A[:, i].nonzero()[0] # indices das linhas em que a coluna i de A não é zero

        if len(row_inds) == 0:
            continue  # continua no caso em que todos os índices são zero

        A_sub = A[row_inds, :][:, row_inds].toarray()
        e_sub = e_i[row_inds]

        # Solve
        m_i, _, _, _ = np.linalg.lstsq(A_sub, e_sub, rcond=None)

        # Populate M
        for idx, val in zip(row_inds, m_i):
            if abs(val) > tol:
                M_rows.append(idx)
                M_cols.append(i)
                M_data.append(val)

    # Make M sparse
    M = sp.csr_matrix((M_data, (M_rows, M_cols)), shape=A.shape)
    return M


if __name__ == "__main__":
    N = 10
    A = sp.diags([4, -1, -1], [0, -1, 1], shape=(N, N), format='csr')
    M = build_spai(A)

    # Check norm
    I_approx = A @ M
    print(I_approx.toarray())
    print(np.linalg.norm(I_approx.toarray() - np.identity(N)))
