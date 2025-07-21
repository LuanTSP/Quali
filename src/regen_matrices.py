from scipy.sparse import csr_matrix, save_npz
from src.build_random_matrix import build_random_matrix
import os


def regen_matrices(dims: list) -> None:
    # cria caminho se não existe
    if not os.path.exists("matrices/"):
        os.makedirs("matrices")
    
    # deleta arquivos anteriores
    dirs = os.listdir("matrices")
    if not dirs.count == 0:
        for filename in dirs:
            filepath = os.path.join("matrices", filename)
            if (os.path.isfile(filepath)):
                os.remove(filepath)
    
    for N in dims:
        M = csr_matrix(build_random_matrix(N=N, m=4))
        save_npz(f"matrices/{N}", M)


if __name__ == "__main__":
    dims = [10, 100, 300, 500, 800, 1000, 2000, 3000, 4000, 5000]
    regen_matrices(dims)
