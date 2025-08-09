from scipy.sparse import load_npz
import networkx as nx
import random
from scipy.sparse import csc_matrix
import numpy as np
import time


def load_matrix(filepath: str):
    """
    Faz o carregamento de uma matrix da memória e a retorna
    """
    M = load_npz(filepath)
    return M


def build_random_matrix(N, m):
    """
    Constroi e retorna uma matriz aleatória a partir de uma grafo de Barabasi Albert
    - Não simétrica
    - Diagonal dominante
    - Quadrada
    """
    G = nx.barabasi_albert_graph(N, m)

    M = nx.to_numpy_array(G)

    # muda os elementos não nulos de M para valores aleatorios e seta a diagonal
    # de forma em que a matriz seja diagonal dominante
    for i in range (0,N):
        k = 0
        for j in range (0,N):
            if M[i,j] != 0: 
                k += 1
                M[i,j] = random.random()
        M[i,i] = k
    
    return csc_matrix(M)


def timeit(func):
    """
    Wrapper para que retorna o tempo que uma função leva para executar.
    """
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        return result, elapsed
    
    return wrapper


def _is_square(A: np.ndarray):
    """
    Checa se uma matriz é quadrada ou não
    """
    N = A.shape[0]
    M = A.shape[1]

    if (N == M) and len(A.shape) == 2:
        return True
    
    return False


def _is_line_diagonal_dominant(A: np.ndarray):
    """
    Checa se uma matriz é diagonal dominante por linhas
    """
    N, M = A.shape
    min_MN = min(N, M)

    for i in range(min_MN):
        aii = abs(A[i,i])
        soma_linha_i = np.sum(np.abs(A[i,:])) - aii

        if aii < soma_linha_i:
            return False
    else:
        return True


def _is_column_diagonal_dominant(A: np.ndarray):
    """
    Checa se uma matriz é diagonal dominante por colunas
    """
    N, M = A.shape
    min_MN = min(N, M)

    for i in range(min_MN):
        aii = abs(A[i,i])
        soma_coluna_i = np.sum(np.abs(A[:,i])) - aii

        if aii < soma_coluna_i:
            return False
    else:
        return True


def _is_positive_definite(A: np.ndarray):
    """
    Checa se uma matriz é definida positiva usando fatoração de Cholesky
    """
    try:
        np.linalg.cholesky(A, upper=True)
        return True
    except:
        return False


def _is_symetric(A: np.ndarray):
    """
    Checa se uma matriz é simétrica
    """
    return bool(np.all((A == A.T) == True))


def _sparcity(A: np.ndarray):
    """
    Retorna a razão de esparcidade de A
    """
    total = A.size
    zero_elements = np.count_nonzero(A == 0)
    return zero_elements / total


def get_matrix_info(A: csc_matrix):
    """
    Obtém informações básicas de uma matriz A
    - dimensao                         (int, int)
    - quadrada                         (bool)
    - diagonal dominante por linhas    (bool)
    - diagonal dominante por colunas   (bool)
    - definida positiva                (bool)
    - simétrica                        (bool)
    - esparcidade                      (float)
    """
    
    # cria dicionário de sucesso em testes
    state = {
        "dimensao": None,
        "quadrada": None,
        "diagonal_dominante_linhas": None,
        "diagonal_dominante_colunas": None,
        "definida_positiva": None,
        "simetrica": None,
        "esparcidade": None,
    }
    A = A.toarray()

    # sava a dimensao
    state["dimensao"] = f"{A.shape}"

    # checa de A é quadrada
    state["quadrada"] = _is_square(A)
    
    # checa se A é diagonal dominante por linhas
    state["diagonal_dominante_linhas"] = _is_line_diagonal_dominant(A)

    # checa se A é diagonal dominante por colunas
    state["diagonal_dominante_colunas"] = _is_column_diagonal_dominant(A)
    
    # checa se a matriz é definida positiva (somente de A é quadrada)
    if (state["quadrada"] == True):
        state["definida_positiva"] = _is_positive_definite(A)

    # checa se a matriz é simétrica
    state["simetrica"] = _is_symetric(A)

    # obtém a razão de esparcidade
    state["esparcidade"] = round(_sparcity(A), 4)
    
    return state