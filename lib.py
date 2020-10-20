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

    extensao = ('nearest', cv2.BORDER_REPLICATE)
    """Amplia a borda extendendo o último pixel."""

    reflexao = ('reflect', cv2.BORDER_REFLECT)
    """Amplia a borda refletindo os últimos pixels."""

    reflexao_pula_ultimo = ('mirror', cv2.BORDER_REFLECT_101)
    """
    Amplia a borda refletindo os últimos pixels, mas
    sem o último pixel.
    """

    def __str__(self) -> str:
        """
        Nome que aparece na linha de comando.
        """
        return self.name


# # # # # # # # # # # # # #
# Backends de convolução  #

# tipo das funções de backend de convolução
Backend = Callable[[Image, Kernel, Borda], np.ndarray]


def scipy_convolve(input: Image, kernel: Kernel, borda: Borda=Borda.reflexao_pula_ultimo) -> np.ndarray:
    """
    Convolução de uma imagem com um kernel pelo SciPy.

    Parâmetros
    ----------
    input: np.ndarray
        Matriz representando a imagem lida.
    kernel: np.ndarray
        Matriz com o filtro de convolução.
    borda: Borda
        Modo de tratamento das bordas. Padrão: ``Borda.reflexao_pula_ultimo``.

    Retorno
    -------
    out: np.ndarray
        Resultado da convolução, em ponto flutuante.

    Tratamento de borda
    -------------------
    As opções de borda estão descritas na classe ``Borda``.

    Nota
    ----
    A convolução é feita em ponto flutuante.
    """
    # mudança para float
    input = input.astype(np.float64)
    # convolução
    output = ndimage.convolve(input, kernel, mode=borda[0])
    return output


def opencv_convolve(input: Image, kernel: Kernel, borda: Borda=Borda.reflexao_pula_ultimo) -> np.ndarray:
    """
    Convolução de uma imagem com um kernel pelo OpenCV.

    Parâmetros
    ----------
    input: np.ndarray
        Matriz representando a imagem lida.
    kernel: np.ndarray
        Matriz com o filtro de convolução.
    borda: Borda
        Modo de tratamento das bordas. Padrão: ``Borda.reflexao_pula_ultimo``.

    Retorno
    -------
    out: np.ndarray
        Resultado da convolução, em ponto flutuante.

    Tratamento de borda
    -------------------
    As opções de borda estão descritas na classe ``Borda``.

    Nota
    ----
    A convolução é feita em ponto flutuante.
    """
    # flip do kernel para a correlação equivalente
    kernel = cv2.flip(kernel, flipCode=-1)
    # convolução
    output = cv2.filter2D(input, cv2.CV_64F, kernel, borderType=borda[1])
    return output


# # # # # # # # # # # # #
# Operações auxiliares  #

# tipos das função de transformação para imagem,
# que no caso são: `trunca` e `transforma_limites`
Limitador = Callable[[np.ndarray], Image]


def transforma_limites(array: np.ndarray) -> Image:
    """
    Transforma uma matriz em uma imagem, mapeando linearmente
    os valores para a região [0, 255].

    Parâmetros
    ----------
    array: np.ndarray
        Matriz númerica.

    Retorno
    -------
    img: np.ndarray
        Matriz representando uma imagem.
    """
    xmin, xmax = np.min(array), np.max(array)
    y =  255 * (array - xmin) / (xmax - xmin)
    return trunca(y)


def trunca(array: np.ndarray) -> Image:
    """
    Trunca uma matriz para inteiros de 8 bits, de forma
    que o resultado represente uma imagem digital.

    Parâmetros
    ----------
    array: np.ndarray
        Matriz númerica.

    Retorno
    -------
    img: np.ndarray
        Matriz representando uma imagem.
    """
    img: Image = array.astype(np.uint8)
    # indices usando a imagem original
    img[array <= 0] = 0
    img[array >= 255] = 255

    assert img.ndim == 2
    return img


def combina(*arrays: np.ndarray) -> np.ndarray:
    """
    Combinação pela raiz da soma quadrática dos elementos
    dos vetores.

    Parâmetros
    ----------
    arrays: np.ndarray
        Cada um dos vetores a ser combinado.

    Retorno
    -------
    out: np.ndarray
        Resultado combinado.

    Exemplo
    -------
    >>> import numpy as np
    >>>
    >>> a = np.asarray([1, 2])
    >>> b = np.asarray([3, 2])
    >>> combina(a, b)
    array([3.16227766, 2.82842712])
    >>> combina(a, b, a)
    array([3.31662479, 3.46410162])
    """
    total = sum(np.square(array) for array in arrays)
    return np.sqrt(total)
