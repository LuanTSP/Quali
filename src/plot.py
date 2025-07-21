from matplotlib import pyplot as plt
from helper import *
import pandas as pd
from load_matrix import load_matrix
from spai import compute_spai

def make_benchmark_plot(path: str, save_path=None, title=None) -> None:
    df = pd.read_csv(path)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    # Condition Number
    axs[0].plot(df["p"], df["condition number"], c="red")
    axs[0].set_title("Condition number")
    axs[0].set_xlabel("p")
    axs[0].set_ylabel("condition number")

    # Identity difference
    axs[1].plot(df["p"], df["identity diference"], c="blue")
    axs[1].set_title("Identity difference")
    axs[1].set_xlabel("p")
    axs[1].set_ylabel("identity difference")


    # show
    plt.tight_layout()
    fig.suptitle(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()



if __name__ == "__main__":
    # for size in [10, 100, 300, 500, 800, 1000, 2000, 3000, 5000]:
    #     make_benchmark_plot(
    #         path=f"results/{size}.csv", 
    #         save_path=f"results/{size}.png",
    #         title=f"{size}"
    #     )
    
    A = load_matrix("matrices/100.npz")
    S0 = sp.csr_matrix(get_sparcity_pattern(A, p=0))
    S05 = sp.csr_matrix(get_sparcity_pattern(A, p=0.5))
    S1 = sp.csr_matrix(get_sparcity_pattern(A, p=1))

    show_sparcity_pattern([A, S0, S05, S1], ["A", "SPAI-1", "SPAI-0.5", "SPAI-0"])

    
