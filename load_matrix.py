from scipy.sparse import load_npz


def load_matrix(filepath: str):
    M = load_npz(filepath)
    return M


if __name__ == "__main__":
    A = load_matrix("matrices/10x10.npz")
    print(A)