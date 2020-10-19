"""
Operações de processamento de imagens.

Nota
----
Todas as operações aqui fazem cópia da imagem para evitar
alterar inesperadamente o buffer interno dos vetores.
"""
from typing import Optional, Tuple, List, Callable
from enum import Enum, unique
from tipos import Image, Kernel

from scipy import ndimage
import numpy as np
import cv2


# # # # # # # # # # # # #
# Operações auxiliares  #

Transform = Callable[[np.ndarray], Image]

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


@unique
class Borda(Tuple[str, int], Enum):
    EXTENSAO = ('nearest', cv2.BORDER_REPLICATE)
    REFLEXAO = ('reflect', cv2.BORDER_REFLECT)
    REFLEXAOOOOO = ('mirror', cv2.BORDER_REFLECT_101)


def scipy_convolve(input: Image, kernel: Kernel, *, borda: Borda=Borda.REFLEXAO, transforma: Transform=trunca) -> Image:
    input = input.astype(np.float64)
    output = ndimage.convolve(input, kernel, mode=borda[0])
    return transforma(output)


def opencv_convolve(input: Image, kernel: Kernel, *, borda: Borda=Borda.REFLEXAO, transforma: Transform=trunca) -> Image:
    kernel = cv2.flip(kernel, flipCode=-1)
    output = cv2.filter2D(input, cv2.CV_64F, kernel, borderType=borda[1])
    return transforma(output)
