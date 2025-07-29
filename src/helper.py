from matplotlib import pyplot as plt
import scipy.sparse as sp
from scipy.sparse import load_npz
import networkx as nx
import random
from spai import compute_spaip_pattern
from scipy.sparse import csc_matrix
import numpy as np
import pandas as pd
import os


def load_matrix(filepath: str):
    M = load_npz(filepath)
    return M


def build_random_matrix(N, m):
    G = nx.barabasi_albert_graph(N, m)

    M = nx.to_numpy_array(G)

    for i in range (0,N):
        k = 0
        for j in range (0,N):
            if M[i,j] != 0: 
                k += 1
                M[i,j] += 2*random.random() 
        M[i,i] = k  
    
    return csc_matrix(M)


def show_spai_sparsity(A: sp.csc_matrix, p_values=None):
    if p_values is None:
        p_values = np.linspace(0, 2, 6)

    n = len(p_values)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axs = axs.flatten()

    for idx, p in enumerate(p_values):
        pattern = np.array(compute_spaip_pattern(A, p))
        axs[idx].spy(pattern)
        axs[idx].set_title(f"SPAI pattern (p={p:.2f})")
        axs[idx].set_xlabel("Column")
        axs[idx].set_ylabel("Row")

    for i in range(idx + 1, len(axs)):
        for j in range(len[axs[0]]):
            axs[i][j].axis("off")
            axs[i][j].grid("off")

    fig.savefig("plots/sparcity.png")


def plot_spai_results(filename: str = "spai_benchmark.csv"):
    df = pd.read_csv(f"csv/{filename}")
    
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("SPAI Benchmark Results", fontsize=16)

    axs[0, 0].plot(df["p"], df["gmres_iters_no_prec"], label="gmres iters (no prec)", marker='o')
    axs[0, 0].plot(df["p"], df["gmres_iters_with_spai"], label="gmres iters (with SPAI)", marker='s')
    axs[0, 0].set_ylabel("# Iterations")
    axs[0, 0].legend()

    axs[0, 1].plot(df["p"], df["gmres_time_no_prec"], label="gmres time (no prec)", marker='o')
    axs[0, 1].plot(df["p"], df["gmres_time_with_spai"], label="gmres time (with SPAI)", marker='s')
    axs[0, 1].set_ylabel("Time (s)")
    axs[0, 1].legend()

    axs[1, 0].plot(df["p"], df["spai_build_time"], label="SPAI Build Time", color='tab:green', marker='^')
    axs[1, 0].set_ylabel("Time (s)")
    axs[1, 0].legend()

    axs[1, 1].plot(df["p"], df["identity_diff_frobenius"], label="‖MA - I‖_F", color='tab:red', marker='d')
    axs[1, 1].set_ylabel("Frobenius Norm")
    axs[1, 1].legend()

    axs[2, 0].plot(df["p"], df["condition_number_est"].replace("NaN", np.nan).astype(float),
                   label="Condition Number Estimate", color='tab:purple', marker='x')
    axs[2, 0].set_ylabel("Cond. Number")
    axs[2, 0].legend()

    for ax in axs.flat:
        ax.set_xlabel("p")
        ax.grid(True)

    axs[2, 1].axis("off")  # leave the last panel empty

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"plots/{filename.replace(".csv", "")}.png")


def make_plot() -> None:
    print("Making Plot Images...")
    dirs = os.listdir("csv")
    for file in dirs:
        filename = file.replace(".csv", "")
        print(f"Saving image {filename}.png")
        plot_spai_results(file)
    print("Images saved!")