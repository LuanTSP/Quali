from benchmark import benchmark
from visualization import make_plots
from regen_matrices import regen_matrices

def main():
    """
        Produz novamente as matrizes e plots usados
    """
    # Gera matrizes aleatórias de tamanho dado na pasta matrices
    regen_matrices([100])

    # faz um benchmark das matrizes geradas na pasta matrices a sava os dados como csv
    # benchmark()



if __name__ == "__main__":
    main()