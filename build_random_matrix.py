import numpy as np
import networkx as nx
import random

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
    
    return M
    

if __name__ == "__main__":
    M = build_random_matrix(100, 4) 
    print(np.linalg.matrix_rank(M))