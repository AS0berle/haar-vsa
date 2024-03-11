"""
Define bind, unbind, identity, and contains operations
"""

import numpy as np
import time
from functools import cache


def get_rand_unit_vec(dim:int, translation:int = 0):
    """ Get unit vectors randomly sampled from the unit sphere

        Optionally, translate them by a fixed amount in all dimensions.
    """
    s = np.random.normal(0, 1, dim)
    return translation + (s / np.linalg.norm(s))


def cos_sim(x:np.ndarray, y:np.ndarray):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def make_normalization_mat(n:int):
    """ Create diagonal matrix which normalizes 0-1 Haar matrix
    """
    n = int(n)
    if n < 1:
        raise Exception("n too small!")
    values = []
    values.append(2**(-n/2))
    for i in range(n):
        curr_val = 2**(-(n-i)/2)
        values.extend([curr_val] * 2**(i))
    return np.diag(values)

#@cache
def make_zero_one_haar(n:int):
    """ Recursively makes Haar matrix of size 2^n
    
        There's probably a more efficient way, but this was easy to implement
    """
    n = int(n)
    if n < 1:
        raise Exception("n too small!")
    if n == 1:
        return np.array([[1, 1], [1, -1]])
    
    kron_haar = np.array([[1, 1]])
    kron_ident = np.array([[1, -1]])
    return np.concatenate(
        (np.kron(make_zero_one_haar(n-1), kron_haar), 
         np.kron(np.identity(2**(n-1)), kron_ident)), 
        axis=0)


def make_ortho_haar(n:int):
    """ Makes the orthogonal (normalized) Haar matrix
    """
    return np.matmul(make_normalization_mat(n), make_zero_one_haar(n))


def inverse_vec(H: np.ndarray, x: np.ndarray):
    inv = 1 / (H @ x)
    return H.T @ inv


def bind(H:np.ndarray, x: np.ndarray, y: np.ndarray):
    Hx = H @ x
    Hy = H @ y
    return H.T @ (Hx * Hy)


def unbind(H:np.ndarray, x: np.ndarray, B: np.ndarray):
        x_inv = inverse_vec(H, x)
        return bind(H, x_inv, B)


def contains_dot(H:np.ndarray, x: np.ndarray, B: np.ndarray):
    x_inv = inverse_vec(H, x)
    return np.dot(x_inv, B)


def contains_cos_sim(H:np.ndarray, x: np.ndarray, B: np.ndarray):
    x_inv = inverse_vec(H, x)
    return cos_sim(x_inv, B)



if __name__ == '__main__':
    n = 12
    H = make_ortho_haar(n)
    x = get_rand_unit_vec(2**n)
    y = get_rand_unit_vec(2**n)
    z = get_rand_unit_vec(2**n)

    B = bind(H, x, y)
    U = bind(H, y, z)
    print("x =", x)
    print("y =", y)

    print("Contains(True)", contains_cos_sim(H, x, B))
    print("Contains(False)", contains_cos_sim(H, x, U))