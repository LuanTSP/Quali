from matplotlib import pyplot as plt
import scipy.sparse as sp
from scipy.sparse import load_npz
import networkx as nx
import random
from spai import make_spai_pattern
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
        pattern = np.array(make_spai_pattern(A, p))
        axs[idx].spy(pattern)
        axs[idx].set_title(f"SPAI pattern (p={p:.2f})")
        axs[idx].set_xlabel("Column")
        axs[idx].set_ylabel("Row")

    for i in range(idx + 1, len(axs)):
        for j in range(len[axs[0]]):
            axs[i][j].axis("off")
            axs[i][j].grid("off")

    fig.savefig("plots/sparcity.png")


def plot_spai_results(filename: str):
    # lendo csv
    df = pd.read_csv(f"csv/{filename}.csv")
    
    # criar pasta de destino se não existir
    folderpath = os.path.join("plots", filename)
    if not os.path.exists(folderpath):
        os.mkdir(folderpath)

    plt.style.use("seaborn-v0_8-darkgrid")

    # gmres iteration benchmark
    plt.plot(df["p"], df["gmres_iters_no_prec"], label="gmres iters (no prec)", marker='o')
    plt.plot(df["p"], df["gmres_iters_with_spai"], label="gmres iters (with SPAI)", marker='s')
    plt.ylabel("# Iterations")
    plt.title("GMRES iterations")
    plt.xlabel("p")
    plt.legend()
    plt.savefig(os.path.join(folderpath, f"{filename} - gmres iterations"))
    plt.close()

    # gmres solve time
    plt.plot(df["p"], df["gmres_time_no_prec"], label="gmres time (no prec)", marker='o')
    plt.plot(df["p"], df["gmres_time_with_spai"], label="gmres time (with SPAI)", marker='s')
    plt.ylabel("Time (s)")
    plt.title("GMRES solve time")
    plt.xlabel("p")
    plt.legend()
    plt.savefig(os.path.join(folderpath, f"{filename} - gmres solve time"))
    plt.close()

    # spai build time
    plt.plot(df["p"], df["spai_build_time"], label="SPAI Build Time", color='tab:green', marker='^')
    plt.ylabel("Time (s)")
    plt.title("SPAI build time")
    plt.xlabel("p")
    plt.legend()
    plt.savefig(os.path.join(folderpath, f"{filename} - spai build time"))
    plt.close()

    # identity difference
    plt.plot(df["p"], df["identity_diff_frobenius"], label="‖MA - I‖_F", color='tab:red', marker='d')
    plt.ylabel("Frobenius Norm")
    plt.title("Identity difference")
    plt.xlabel("p")
    plt.legend()
    plt.savefig(os.path.join(folderpath, f"{filename} - identity difference"))
    plt.close()

    # condition number
    plt.plot(df["p"], df["condition_number_est"].replace("NaN", np.nan).astype(float), label="Condition Number Estimate", color='tab:purple', marker='x')
    plt.ylabel("Cond. Number")
    plt.title("Condition number")
    plt.xlabel("p")
    plt.legend()
    plt.savefig(os.path.join(folderpath, f"{filename} - condition number"))
    plt.close()



def make_plot() -> None:
    print("Making Plot Images...")
    dirs = os.listdir("csv")
    for file in dirs:
        print(f"Making plots for {file}")
        filename = file.replace(".csv", "")
        plot_spai_results(filename)
    print("Images saved!")


def plot_D_A_A2(A, save=False):
    # Make plot of D, A, A^2
    fig, axs = plt.subplots(nrows=1, ncols=3)

    SP0 = make_spai_pattern(A, 0)
    axs[0].spy(SP0)
    axs[0].set_title("Matriz diagonal")
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    axs[1].spy(A)
    axs[1].set_title("A")
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    axs[2].spy(A @ A)
    axs[2].set_title("A * A")
    axs[2].set_xticks([])
    axs[2].set_yticks([])

    fig.suptitle("Matrizes que devem ser interpoladas:")

    if save:
        plt.savefig("plots/matrizes_D_A_A2.png")

    plt.show()


def plot_spai_interpolation(A, save=False):
    # Make transition plots:
    fig, axs = plt.subplots(nrows=2, ncols=4)

    # Linha 0
    axs[0][0].spy(make_spai_pattern(A, 0))
    axs[0][1].spy(make_spai_pattern(A, 0))
    axs[0][2].spy(make_spai_pattern(A, 0.2))
    axs[0][3].spy(make_spai_pattern(A, 0.4))
    axs[1][0].spy(make_spai_pattern(A, 0.6))
    axs[1][1].spy(make_spai_pattern(A, 0.8))
    axs[1][2].spy(make_spai_pattern(A, 1))
    axs[1][3].spy(A)

    titles = [["D", "p=0", "p=0.2", "p=0.4"], ["p=0.6", "p=0.8", "p=1", "A"]]
    for i in range(2):
        for j in range(4):
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
            title = titles[i][j]
            axs[i][j].set_title(title)
    

    fig.suptitle("Interpolation")
    
    if save:
        plt.savefig("plots/Interpolation [0,1].png")
    
    plt.show()

    # Make transition plots:
    fig, axs = plt.subplots(nrows=2, ncols=4)

    # Linha 0
    axs[0][0].spy(A)
    axs[0][1].spy(make_spai_pattern(A, 1))
    axs[0][2].spy(make_spai_pattern(A, 1.2))
    axs[0][3].spy(make_spai_pattern(A, 1.4))
    axs[1][0].spy(make_spai_pattern(A, 1.6))
    axs[1][1].spy(make_spai_pattern(A, 1.8))
    axs[1][2].spy(make_spai_pattern(A, 2))
    axs[1][3].spy(A @ A)

    titles = [["A", "p=1", "p=1.2", "p=1.4"], ["p=1.6", "p=1.8", "p=2", "A * A"]]
    for i in range(2):
        for j in range(4):
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
            title = titles[i][j]
            axs[i][j].set_title(title)
    

    fig.suptitle("Interpolation")
    
    if save:
        plt.savefig("plots/Interpolation [1,2].png")

    plt.show()