import pandas as pd
from spai import *
from helper import *
import os
import time
from scipy.sparse.linalg import eigsh, bicgstab

class Counter:
    def __init__(self):
        self.iters = 0
    
    def __call__(self, xk=None):
        self.iters += 1


def benchmark_spai(A: sp.csc_matrix, b: np.ndarray, filename: str = "spai_benchmark.csv", p_values=None):
    tol = 1e-6
    maxiter = 2 * A.shape[0]
    
    if p_values is None:
        p_values = np.linspace(0, 2, 11)

    records = []

    # Resolver sem o precondicionador (tempo)
    counter1 = Counter()
    t0 = time.time()
    x0, code0 = bicgstab(A, b, rtol=tol, callback=counter1, maxiter=maxiter)
    t1 = time.time()

    if (code0 > 0):
        raise RuntimeError(f"Error: '{code0}' Iterative solver did not converge for atol={tol}, maxiter={maxiter}")
        
    time_no_prec = t1 - t0
    iters_no_prec = counter1.iters

    for p in p_values:
        print(f"Benchmarking SPAI for p = {p:.2f}")

        # Construir precondicionador (tempo)
        t2 = time.time()
        M = fast_spai(A, p)
        t3 = time.time()
        spai_time = t3 - t2

        # Define aplicação do precondicionador
        def apply_M(x):
            return M @ x

        M_linop = sp.linalg.LinearOperator(A.shape, matvec=apply_M)

        # Resolve com o precondicionador (tempo)
        counter2 = Counter()
        t4 = time.time()
        x1, code1 = bicgstab(A, b, M=M_linop, rtol=tol, callback=counter2, maxiter=maxiter)
        t5 = time.time()

        if (code1 > 0):
            raise RuntimeError(f"Error: '{code1}' Iterative solver did not converge for atol={tol}, maxiter={maxiter}")

        time_with_prec = t5 - t4
        iters_with_prec = counter2.iters

        # Deferenca da identidade de M @ A
        B = M @ A
        identity_diff = sp.linalg.norm(B - sp.eye(A.shape[0]), ord='fro')

        # Estima o numero de condicao
        try:
            evals = eigsh(B, k=2, return_eigenvectors=False, which='BE')
            cond_num = abs(evals[1]) / abs(evals[0])
        except:
            cond_num = np.nan

        # Salva os dados
        records.append({
            "p": round(p, 4),
            "gmres_iters_no_prec": iters_no_prec,
            "gmres_time_no_prec": round(time_no_prec, 6),
            "gmres_iters_with_spai": iters_with_prec,
            "gmres_time_with_spai": round(time_with_prec, 6),
            "spai_build_time": round(spai_time, 6),
            "identity_diff_frobenius": round(identity_diff, 6),
            "condition_number_est": round(cond_num, 6) if not np.isnan(cond_num) else "NaN"
        })

    df = pd.DataFrame(records)
    df.to_csv(filename, index=False)
    print(f"Benchmark results saved to {filename}")


def benchmark() -> None:
    dirs = os.listdir("matrices")
    for file in dirs:
        A = load_matrix(os.path.join("matrices", file))
        b = np.random.rand(A.shape[0])
        print(f"SPAI-p SIZE {A.shape[0]} BENCHMARK:")
        benchmark_spai(A, b, filename=f"csv/{A.shape[0]}.csv")
    print("Benchmark concluded")