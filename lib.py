"""
Operações de processamento de imagens.

Nota
----
Todas as operações aqui fazem cópia da imagem para evitar
alterar inesperadamente o buffer interno dos vetores.
"""
from __future__ import annotations
from typing import Optional, Tuple, List, Callable, TYPE_CHECKING
from enum import Enum, unique
from tipos import Image, Kernel

import cv2
import numpy as np
from scipy import ndimage


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


if TYPE_CHECKING:
    Opcoes = Tuple[str, int]
else:
    Opcoes = tuple

@unique
class Borda(Opcoes, Enum):
    extensao = ('nearest', cv2.BORDER_REPLICATE)
    reflexao = ('reflect', cv2.BORDER_REFLECT)
    reflexaooo = ('mirror', cv2.BORDER_REFLECT_101)

    # https://gist.github.com/ptmcg/23ba6e42d51711da44ba1216c53af4ea
    @classmethod
    def argtype(cls, opcao: str) -> Borda:
        try:
            return cls[opcao]
        except KeyError:
            ...

    def __str__(self) -> str:
        return self.name



Backend = Callable[[Image, Kernel, Borda], np.ndarray]

def scipy_convolve(input: Image, kernel: Kernel, borda: Borda=Borda.reflexao) -> np.ndarray:
    input = input.astype(np.float64)
    output = ndimage.convolve(input, kernel, mode=borda[0])
    return output


def opencv_convolve(input: Image, kernel: Kernel, borda: Borda=Borda.reflexao) -> np.ndarray:
    kernel = cv2.flip(kernel, flipCode=-1)
    output = cv2.filter2D(input, cv2.CV_64F, kernel, borderType=borda[1])
    return output
