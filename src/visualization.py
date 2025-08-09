import numpy as np
from spai import make_spai_pattern
from matplotlib import pyplot as plt
import scipy.sparse as sp
import os
import pandas as pd


def save_img_multiple_spaip_sparsities(A: sp.csc_matrix, p_values=None):
    """
    Saves a figure of the sparcity pattern os SPAI-p for vaoius values of p
    inside the 'plots' directory.
    """
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


def plot_spai_results(csv_path: str):
    # lendo csv
    df = pd.read_csv(csv_path)
    parent = os.path.dirname(csv_path)

    plt.style.use("seaborn-v0_8-darkgrid")

    # iteration benchmark
    plt.plot(df["p"], df["iters_no_prec"], label=" iters (no prec)", marker='o')
    plt.plot(df["p"], df["iters_with_prec"], label=" iters (with SPAIp)", marker='s')
    plt.ylabel("# Iterations")
    plt.title("iterations count")
    plt.xlabel("p")
    plt.legend()
    plt.savefig(os.path.join(parent, "iterations count"))
    plt.close()

    #  solve time
    plt.plot(df["p"], df["time_no_prec"], label="time (no prec)", marker='o')
    plt.plot(df["p"], df["time_with_prec"], label="time (with SPAI)", marker='s')
    plt.ylabel("Time (s)")
    plt.title("solve time")
    plt.xlabel("p")
    plt.legend()
    plt.savefig(os.path.join(parent, "solve time"))
    plt.close()

    # spai build time
    plt.plot(df["p"], df["spai_build_time"], label="SPAIp Build Time", color='tab:green', marker='^')
    plt.ylabel("Time (s)")
    plt.title("SPAIp build time")
    plt.xlabel("p")
    plt.legend()
    plt.savefig(os.path.join(parent, "spaip build time"))
    plt.close()

    # identity difference
    plt.plot(df["p"], df["identity_diff_frobenius"], label="‖MA - I‖_F", color='tab:red', marker='d')
    plt.ylabel("Frobenius Norm")
    plt.title("Identity difference")
    plt.xlabel("p")
    plt.legend()
    plt.savefig(os.path.join(parent, "identity difference"))
    plt.close()

    # condition number
    plt.plot(df["p"], df["condition_number_est"].replace("NaN", np.nan).astype(float), label="Condition Number Estimate", color='tab:purple', marker='x')
    plt.ylabel("Cond. Number")
    plt.title("Condition number")
    plt.xlabel("p")
    plt.legend()
    plt.savefig(os.path.join(parent, "condition number"))
    plt.close()


def make_plots() -> None:
    """
    Plots the results for all csv data inside the 'csv' directory inside the 'plots' directory
    """
    print("Making Plot Images...")
    dirs = os.listdir("csv")
    for file in dirs:
        print(f"Making plots for {file}")
        filename = file.replace(".csv", "")
        plot_spai_results(filename)
    print("Images saved!")


def plot_spai_interpolation(A, save=False):
    """
    Shows the interpolation plot for spaip for various values of p
    """
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