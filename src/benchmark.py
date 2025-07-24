import pandas as pd
from spai import *
from helper import *
import os
import time
from scipy.sparse.linalg import cg, eigsh


def benchmark_spai(A: sp.csc_matrix, b: np.ndarray, filename: str = "spai_benchmark.csv", p_values=None):
    if p_values is None:
        p_values = np.linspace(0, 2, 11)

    records = []

    for p in p_values:
        print(f"Benchmarking SPAI for p = {p:.2f}")

        # Solve without preconditioner
        t0 = time.time()
        x0, info0 = cg(A, b, atol=1e-6)
        t1 = time.time()
        time_no_prec = t1 - t0
        iters_no_prec = len(x0) if info0 == 0 else -1

        # Build SPAI preconditioner
        t2 = time.time()
        M = fast_spai(A, p)
        t3 = time.time()
        spai_time = t3 - t2

        # Define preconditioner application
        def apply_M(x):
            return M @ x

        M_linop = sp.linalg.LinearOperator(A.shape, matvec=apply_M)

        # Solve with preconditioner
        t4 = time.time()
        x1, info1 = cg(A, b, M=M_linop, atol=1e-6)
        t5 = time.time()
        time_with_prec = t5 - t4
        iters_with_prec = len(x1) if info1 == 0 else -1

        # Benchmark preconditioned system B = M @ A
        B = M @ A
        identity_diff = sp.linalg.norm(B - sp.eye(A.shape[0]), ord='fro')

        # Estimate condition number using largest and smallest eigenvalues
        try:
            evals = eigsh(B, k=2, return_eigenvectors=False, which='BE')
            cond_num = abs(evals[1]) / abs(evals[0])
        except:
            cond_num = np.nan

        records.append({
            "p": round(p, 4),
            "cg_iters_no_prec": iters_no_prec,
            "cg_time_no_prec": round(time_no_prec, 6),
            "cg_iters_with_spai": iters_with_prec,
            "cg_time_with_spai": round(time_with_prec, 6),
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
        b = np.zeros(A.shape[0])
        print(f"SPAI-p SIZE {A.shape[0]} BENCHMARK:")
        benchmark_spai(A, b, filename=f"csv/{A.shape[0]}.csv")
    print("Benchmark concluded")