from benchmark import benchmark
from helper import *
from spai import *
from regen_matrices import regen_matrices

def main():
    """
        Produz novamente as matrizes e plots usados
    """
    # Gera matrizes aleatórias de tamanho dado na pasta matrices
    regen_matrices([15, 100, 500, 1000, 2000, 3000, 4000, 5000])

    # faz um benchmark das matrizes geradas na pasta matrices a sava os dados como
    # um arquivo .csv na pasta csv
    benchmark()

    # salva um gráfico para cada resultado .csv salvo na pasta plots
    make_plot()
    
    # sava um gráfico do padrão de esparcidade de A na pasta plots
    A = load_matrix("matrices/15.npz").toarray()
    plot_D_A_A2(A, save=True)
    plot_spai_interpolation(A, save=True)



if __name__ == "__main__":
    main()