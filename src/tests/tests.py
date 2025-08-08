from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigs
import numpy as np
from helper import build_random_matrix, timeit
from spai import fast_spai, make_spai_0

def validade_spaip_at_0(N: int, spaip):
    """
    Retorna True se fast_spai retorna
    """
    if N < 1:
        raise ValueError("N é um valor inteiro positivo")
    
    A = build_random_matrix(N, 4)
    N = A.shape[0]
    
    D = make_spai_0(A).toarray()
    D2 = spaip(A, 0).toarray()
    
    if np.sum(np.isclose(D, D2, atol=1e-6)) != N * N:
        return False
    
    return True


def is_square(A: np.ndarray):
    """
    Checa se uma matriz é quadrada ou não
    """
    N = A.shape[0]
    M = A.shape[1]

    if (N == M) and len(A.shape) == 2:
        return True
    
    return False

def is_line_diagonal_dominant(A: np.ndarray):
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


def is_column_diagonal_dominant(A: np.ndarray):
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


def is_positive_definite(A: np.ndarray):
    """
    Checa se uma matriz é definida positiva usando fatoração de Cholesky
    """
    try:
        np.linalg.cholesky(A, upper=True)
        return True
    except:
        return False


def is_symetric(A: np.ndarray):
    """
    Checa se uma matriz é simétrica
    """
    return bool(np.all((A == A.T) == True))


def sparcity(A: np.ndarray):
    """
    Retorna a razão de esparcidade de A
    """
    total = A.size
    zero_elements = np.count_nonzero(A == 0)
    return zero_elements / total


def get_info(A: csc_matrix):
    """
    Obtém informações básicas de uma matriz A
    - quadrada                         (bool)
    - diagonal dominante por linhas    (bool)
    - diagonal dominante por colunas   (bool)
    - definida positiva                (bool)
    - simétrica                        (bool)
    - esparcidade                      (float)
    """
    
    # cria dicionário de sucesso em testes
    state = {
        "quadrada": None,
        "diagonal_dominante_linhas": None,
        "diagonal_dominante_colunas": None,
        "definida_positiva": None,
        "simetrica": None,
        "esparcidade": None,
    }
    A = A.toarray()

    # checa de A é quadrada
    state["quadrada"] = is_square(A)
    
    # checa se A é diagonal dominante por linhas
    state["diagonal_dominante_linhas"] = is_line_diagonal_dominant(A)

    # checa se A é diagonal dominante por colunas
    state["diagonal_dominante_colunas"] = is_column_diagonal_dominant(A)
    
    # checa se a matriz é definida positiva (somente de A é quadrada)
    if (state["quadrada"] == True):
        state["definida_positiva"] = is_positive_definite(A)

    # checa se a matriz é simétrica
    state["simetrica"] = is_symetric(A)

    # obtém a razão de esparcidade
    state["espacidade"] = round(sparcity(A), 4)
    
    return state