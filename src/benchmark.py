import os
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from spai import fast_spai
from helper import timeit, load_matrix, get_matrix_info
from scipy.sparse.linalg import eigsh, bicgstab, LinearOperator
from visualization import plot_spai_results

class Counter:
    def __init__(self):
        self.iters = 0
    
    def __call__(self, xk=None):
        self.iters += 1


@timeit
def solve_without_preconditioner(A: sp.csc_matrix, b: np.ndarray, tol: float, maxiter: int, solver):
    """
    Resolve o sistema sem precondicionadores
    Retorna:
        - x     (solução)
        - code  (return code)
        - iters (number of iterations)
        - time  (time to solve)
    """

    N = A.shape[0]
    print(f"solving system of size {N} without preconditioner ...")
    
    # solving and counting iterations
    counter1 = Counter()
    x, code = solver(A, b, rtol=tol, callback=counter1, maxiter=maxiter)

    if (code > 0):
        print(f"\nWARNING: '{code}' Iterative solver did not converge for atol={tol}, maxiter={maxiter}\n")

    iters = counter1.iters
    return x, code, iters


@timeit
def solve_with_preconditioner(A: sp.csc_array, b: np.ndarray, M: LinearOperator, tol: float, maxiter: int, solver):
    """
    Resolve o sistema com precondicionador M
    Retorna:
        - x     (solução)
        - code  (return code)
        - iters (number of iterations)
        - time  (time to solve)
    """
    
    counter = Counter()
    x, code = solver(A, b, M=M, rtol=tol, callback=counter, maxiter=maxiter)

    if (code > 0):
        print(f"\nWARNING: '{code}' Iterative solver did not converge for atol={tol}, maxiter={maxiter}\n")

    iters = counter.iters
    return x, code, iters


def benchmark_spaip(A: sp.csc_matrix, b: np.ndarray, tol: float=1e-6, maxiter: int=500, solver=bicgstab):
    p_values = np.linspace(0, 2, 11)
    N = A.shape[0]
    records = []
    
    # Resolve sem precondicionador
    (_, _, iters_no_prec), time_no_prec = solve_without_preconditioner(A, b, tol, maxiter, solver)

    for p in p_values:
        print(f"benchmarking spai-{round(p,1)} of size {N}")

        # Construir precondicionador (tempo)
        (M), build_time = fast_spai(A, p)

        # Define aplicação do precondicionador
        def apply_M(x):
            return M @ x

        M_linop = sp.linalg.LinearOperator(A.shape, matvec=apply_M)

        # Resolve com o precondicionador (tempo)
        (_, _, iters_with_prec), time_with_prec = solve_with_preconditioner(A, b, M_linop, tol, maxiter, solver)

        # Deferenca da identidade de M @ A
        B = M @ A
        identity_diff = sp.linalg.norm(B - sp.eye(A.shape[0]), ord='fro')

        # Estima o numero de condicao
        try:
            evals = eigsh(B, k=2, return_eigenvectors=False, which='BE')
            cond_num = abs(evals[1]) / abs(evals[0])
        except:
            cond_num = np.nan

        # guardas os dados para p
        records.append({
            "p": round(p, 4),
            "iters_no_prec": iters_no_prec,
            "time_no_prec": round(time_no_prec, 6),
            "iters_with_prec": iters_with_prec,
            "time_with_prec": round(time_with_prec, 6),
            "spai_build_time": round(build_time, 6),
            "identity_diff_frobenius": round(identity_diff, 6),
            "condition_number_est": round(cond_num, 6) if not np.isnan(cond_num) else "NaN"
        })

    # cria a pasta benchmarks se não existir
    folder = "benchmarks"
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    # limpa os dados anteriores das subpastas dentro de 'folder' se existirem
    folder_N = os.path.join(folder, f"{N}")
    if not os.path.exists(folder_N):
        os.mkdir(folder_N)
    else:
        for file in os.listdir(folder_N):
            filepath = os.path.join(folder_N, file)
            os.remove(filepath)
    
    # sava os dados em 'records' em um arquivo csv 
    csv_path = os.path.join(folder_N, f"{N}.csv")
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"benchmark results saved to {csv_path}")

    # faz os plots necessários
    print(f"saving plots for spai-{round(p, 1)}")
    plot_spai_results(csv_path)

    # faz uma log das qualidade da matriz A
    matrix_info_path = os.path.join(folder_N, f"{N}.txt")
    with open(matrix_info_path, "x") as f:
        header = "##################################################\nMATRIX PROPERTIES\n##################################################\n\n"
        
        lines = [header]
        
        info = get_matrix_info(A)
        for key in info:
            lines.append(f"{key} : {info[key]}\n")

        f.writelines(lines)
        f.close()

 

def benchmark() -> None:
    dirs = os.listdir("matrices")
    for file in dirs:
        A = load_matrix(os.path.join("matrices", file))
        b = np.random.rand(A.shape[0])
        print(f"\nspai-p SIZE {A.shape[0]} benchmark:\n")
        benchmark_spaip(A=A, b=b)
    print("\nBenchmark concluded!\n")