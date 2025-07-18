import os
import pandas as pd
import numpy as np
from load_matrix import load_matrix
from build_spai import build_spai
from build_spai_p01_matrix import build_spai_p01_matrix
from scipy.sparse.linalg import svds

def conditionN(A):
    try:
        s = svds(A, k=2, return_singular_vectors=False)

        sigma_min = min(s)
        sigma_max = max(s)
    except:
        return "inf"


    return sigma_max / sigma_min


def benchmark(path: str) -> None:
    # Checar se diretório existe
    if not os.path.exists(path):
        raise FileExistsError("'matrices' path does not exist, please run 'regen_matrices.py first'")

    dirs = os.listdir(path)
    data = []
    for file in dirs:
        filepath = os.path.join(path, file)
        
        # Contruir matrizes
        A = load_matrix(filepath=filepath)
        spai = build_spai(A)
        spai01 = build_spai_p01_matrix(A, 0.2)
        
        # Métricas
        IdN = np.identity(A.shape[0])
        data.append([
            file, 
            conditionN(A), conditionN(A @ spai), conditionN(A @ spai01), #cond
            np.linalg.norm(A - IdN), np.linalg.norm(A @ spai - IdN), np.linalg.norm(A @ spai01 - IdN),
        ])

    
    # End of benchmark
    df = pd.DataFrame(data, columns=[
        "Size", 
        "A (cond)", "A * SPAI (cond)", "A * SPAI-p (cond)",
        "A - Id (norm)", "A * SPAI - Id (norm)", "A * SPAI-p - Id (norm)"
    ])
    
    if not os.path.exists("results"):
        os.makedirs("results")
    

    df.to_csv("results/results.csv", index=False)
    print("Benchmark concluded!")



if __name__ == "__main__":
    benchmark("matrices")
