import networkx as nx
import matplotlib.pyplot as plt
from spai import make_spai_pattern
from helper import load_matrix
from matplotlib import pyplot as plt
from numpy.linalg import inv
from numpy import abs as npabs

def function1():
        A = load_matrix("matrices/30.npz")

        fig, axs = plt.subplots(nrows=1, ncols=4)

        spai0 = make_spai_pattern(A, 0)
        spai1 = make_spai_pattern(A, 1)
        spai2 = make_spai_pattern(A, 2)

        axs[0].spy(A.toarray())
        axs[1].spy(spai0)
        axs[2].spy(spai1)
        axs[3].spy(spai2)

        names = ["A pattern", "SPAI-0", "SPAI-1", "SPAI-2"]

        for ax, name in zip(axs, names):
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(name)




        fig.suptitle("SPAI patterns")

        plt.show()


A = load_matrix("matrices/100.npz")
A = npabs(A.toarray())

A_inv = inv(A)

A = (A != 0).astype(int)
A2 = (A @ A != 0).astype(int)
A3 = (A @ A @ A != 0).astype(int)
A4 = (A @ A @ A @ A != 0).astype(int)
A_inv = (A_inv != 0).astype(int)

fig, axs = plt.subplots(nrows=1, ncols=5)

axs[0].spy(A_inv)
axs[1].spy(A)
axs[2].spy((A - A2 != 0).astype(int))
axs[3].spy((A - A2 - A3 != 0).astype(int))
axs[4].spy((A - A2 - A3 - A4 != 0).astype(int))


names = ["A^-1", "m=1", "m=2", "m=3", "m=4"]

for ax, name in zip(axs, names):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name)




fig.suptitle("SPAI patterns")

plt.show()

