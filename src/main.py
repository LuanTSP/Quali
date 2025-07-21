from benchmark import benchmark
from helper import *
from build_random_matrix import build_random_matrix
from spai import compute_spai

def main():
    N = 10
    m = 4
    A = build_random_matrix(N, m)
    # p = 0 (same sparcity as A)
    p = 0
    M1 = compute_spai(A, p, 1e-6) # Compute spai 'p' preconditioner from 'A' with sparcity pattern give by 'S'
    SM1 = get_sparcity_pattern(M1, 0)
    benchmark_identity_diference(A, A @ M1)
    benchmark_condition_number(A, A @ M1)

    # p = 0.5 (some vertices removed)
    p = 0.5
    M2 = compute_spai(A, p, 1e-6) # Compute spai 'p' preconditioner from 'A' with sparcity pattern give by 'S'
    SM2 = get_sparcity_pattern(M2, 0)
    benchmark_identity_diference(A, A @ M2)
    benchmark_condition_number(A, A @ M2)

    show_sparcity_pattern([SM1, SM2])




if __name__ == "__main__":
    main()