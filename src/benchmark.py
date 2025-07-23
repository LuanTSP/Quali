import pandas as pd
from spai import *
from helper import *
import os


def benchmark(path: str) -> None:
    A = load_matrix(path)

    P = [_/10 for _ in range(21)]
    data = {
        "p": [],
        "condition number": [],
        "identity diference": [],
    }

    for p in P:
        # calculate spaip preconditioner
        print(f"calculating spai-{p} of size {A.shape[0]} ...")
        AM = A @ spaip(A, p, 1e-6)
        cond = condition_number(AM)
        idiff = identity_difference(AM)

        data["p"].append(p)
        data["condition number"].append(cond)
        data["identity diference"].append(idiff)
    
    df = pd.DataFrame(data, index=None)
    df.to_csv(f"results/{A.shape[0]}.csv")
    print(f"Results saved in file {f"results/{A.shape[0]}.csv"}")
        


if __name__ == "__main__":
    dirs = os.listdir("matrices")
    for file in dirs:
        benchmark(os.path.join("matrices", file))
    print("Benchmark concluded")
