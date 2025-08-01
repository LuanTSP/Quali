from scipy.sparse import csc_matrix, save_npz
from helper import build_random_matrix
import os


def regen_matrices(dims: list) -> None:
    # cria caminho se não existe
    if not os.path.exists("matrices/"):
        os.makedirs("matrices")
    
    # deleta arquivos anteriores
    dirs = os.listdir("matrices")
    print("Generating matrices...")
    if not dirs.count == 0:
        for filename in dirs:
            filepath = os.path.join("matrices", filename)
            if (os.path.isfile(filepath)):
                os.remove(filepath)
    
    for N in dims:
        M = csc_matrix(build_random_matrix(N=N, m=4))
        save_npz(f"matrices/{N}", M)
        print(f"matrix of size {N} generated")
    print("files saves in 'matrices'")


if __name__ == "__main__":
    dims = [10, 100, 300, 500, 800, 1000, 2000, 3000, 4000, 5000]
    regen_matrices(dims)
