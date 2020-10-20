"""
Operações de convolução de imagens e operações auxiliares.

Nota
----
Todas as operações aqui fazem cópia da imagem para evitar
alterações inesperadas do buffer interno dos vetores.
"""
from typing import Tuple, Callable, TYPE_CHECKING
from enum import Enum, unique
from tipos import Image, Kernel

import numpy as np
from scipy import ndimage
import cv2


# # # # # # # # # # # # #
# Tratamento das bordas #

# heranças diferentes para tipagem
# estática e execução dinâmica
if TYPE_CHECKING:
    Opcoes = Tuple[str, int]
else:
    # em Python 3.9 isso não é mais
    # necessário \o/
    Opcoes = tuple

@unique
class Borda(Opcoes, Enum):
    """
    Opções de tratamento das bordas da imagem.
    """
    # entesão do último pixel
    extensao = ('nearest', cv2.BORDER_REPLICATE)
    # reflexão dos pixels da borda
    reflexao = ('reflect', cv2.BORDER_REFLECT)
    # reflexão dos pixels da borda, mas
    # sem o refletir o útlimo pixel
    reflexao_pula_último = ('mirror', cv2.BORDER_REFLECT_101)

    def __str__(self) -> str:
        """
        Nome que aparece na linha de comando.
        """
        return self.name


# # # # # # # # # # # # # #
# Backends de convolução  #

# tipo das funções de backend de convolução
Backend = Callable[[Image, Kernel, Borda], np.ndarray]


def scipy_convolve(input: Image, kernel: Kernel, borda: Borda=Borda.reflexao_pula_um) -> np.ndarray:
    """
    Convolução de uma imagem com um kernel pelo SciPy.

    Borda
    -----
    As opções de borda estão

    Nota
    ----
    A imagem é tratada em ponto flutuante. O retorno da função também
    é em ponto flutuante.
    """
    input = input.astype(np.float64)
    output = ndimage.convolve(input, kernel, mode=borda[0])
    return output


def opencv_convolve(input: Image, kernel: Kernel, borda: Borda=Borda.reflexao_pula_um) -> np.ndarray:
    """
    Convolução de uma imagem com um kernel pelo SciPy.
    """
    kernel = cv2.flip(kernel, flipCode=-1)
    output = cv2.filter2D(input, cv2.CV_64F, kernel, borderType=borda[1])
    return output


# # # # # # # # # # # # #
# Operações auxiliares  #

def combina(*arrays: np.ndarray) -> np.ndarray:
    total = sum(np.square(array) for array in arrays)
    return np.sqrt(total)


Limitador = Callable[[np.ndarray], Image]

def transforma_limites(array: np.ndarray) -> Image:
    """
    Transforma os elementos de array linearmente para a região [0, 255].
    """
    xmin, xmax = np.min(array), np.max(array)
    y =  255 * (array / (xmax - xmin))
    return trunca(y)


def trunca(array: np.ndarray) -> Image:
    """
    Trunca um array qualquer para inteiros de 8 bits, representando uma imagem.
    """
    img: Image = array.astype(np.uint8)
    img[array <= 0] = 0
    img[array >= 255] = 255

    assert img.ndim == 2
    return img
