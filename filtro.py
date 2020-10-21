"""
Definição dos filtros para o trabalho 1.
"""
import numpy as np
from typing import Dict

# anotações de tipo para 3.7+
from sys import version_info
if version_info.minor >= 7:
    from tipos import Kernel
else: # para 3.6
    Kernel = "Kernel"



# # # # # # # # # # # #
# Filtros do trabalho #

h1 = np.asarray([
    [ 0,  0, -1,  0,  0],
    [ 0, -1, -2, -1,  0],
    [-1, -2, 16, -2, -1],
    [ 0, -1, -2, -1,  0],
    [ 0,  0, -1,  0,  0]
])

h2 = (1 / 256) * np.asarray([
    [1,  4,  6,  4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1,  4,  6,  4, 1]
])

h3 = np.asarray([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

h4 = np.asarray([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])

h5 = np.asarray([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

h6 = (1 / 9) * np.asarray([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])

h7 = np.asarray([
    [-1, -1,  2],
    [-1,  2, -1],
    [ 2, -1, -1]
])

h8 = np.asarray([
    [ 2, -1, -1],
    [-1,  2, -1],
    [-1, -1,  2]
])

h9 = (1 / 9) * np.diag(np.ones(9))

h10 = (1 / 8) * np.asarray([
    [-1, -1, -1, -1, -1],
    [-1,  2,  2,  2, -1],
    [-1,  2,  8,  2, -1],
    [-1,  2,  2,  2, -1],
    [-1, -1, -1, -1, -1]
])

h11 = np.asarray([
    [-1, -1, 0],
    [-1,  0, 1],
    [ 0,  1, 1]
])


# # # # # # # # # #
# Acesso por nome #

# Dicinário para acesso dos filtros por nome.
FILTRO: Dict[str, Kernel] = {
    f'h{i}': globals()[f'h{i}']
    for i in range(1, 11+1)
}
