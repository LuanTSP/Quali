from benchmark import benchmark
from helper import *
from spai import *
from regen_matrices import regen_matrices

def main():
    """
        Produz novamente as matrizes e plots usados
    """
    # Gera matrizes aleatórias de tamanho dado na pasta matrices
    regen_matrices([10, 100, 500, 1000, 2000, 3000, 4000, 5000])

    # faz um benchmark das matrizes geradas na pasta matrices a sava os dados como
    # um arquivo .csv na pasta results
    benchmark()

    # salva um gráfico para cada resultado .csv salvo na pasta results
    make_plot()
    
    # sava um gráfico do padrão de esparcidade de A na pasta results
    A = load_matrix("matrices/10.npz")
    show_spai_sparsity(A, p_values=[0, 0.2, 0.4, 0.8, 1, 1.6, 1.8, 2])



if __name__ == "__main__":
    main()