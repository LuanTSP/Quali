from spai import main_spai, fast_spai, parallel_spai, parallel_spai_fast, main_spai_tf
from helper import build_random_matrix
from scipy.sparse.linalg import norm
from scipy.sparse import eye

if __name__ == "__main__":
    A = build_random_matrix(5000, 4)

    print("========= main_spai =========")
    
    M, elapsed = main_spai(A, p=1)
    print(f"{elapsed}s for the whole function")
    I = eye(A.shape[0])
    print(norm(A - I))
    print(norm(M @ A - I))

    print("========= fast_spai =========")

    M, elapsed = fast_spai(A, p=1)
    print(f"{elapsed}s for the whole function")
    I = eye(A.shape[0])
    print(norm(A - I))
    print(norm(M @ A - I))

    print("========= parallel_spai =========")

    M, elapsed = parallel_spai(A, p=1)
    print(f"{elapsed}s for the whole function")
    I = eye(A.shape[0])
    print(norm(A - I))
    print(norm(M @ A - I))

    print("========= parallel_spai_fast =========")

    M, elapsed = parallel_spai_fast(A, p=1)
    print(f"{elapsed}s for the whole function")
    I = eye(A.shape[0])
    print(norm(A - I))
    print(norm(M @ A - I))

    print("========= main_spai_tf =========")

    M, elapsed = main_spai_tf(A, p=1)
    print(f"{elapsed}s for the whole function")
    I = eye(A.shape[0])
    print(norm(A - I))
    print(norm(M @ A - I))

    
