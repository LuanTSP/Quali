import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import random
from scipy.sparse import csr_matrix

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
    
    return csr_matrix(M)
    

if __name__ == "__main__":
    from src.helper import save_sparcity_figure
    # make an image for testing purposes
    M = build_random_matrix(100, 4).toarray()
    save_sparcity_figure(M, "debug/random_sparcity.png")
    
