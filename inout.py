"""
Funções de IO com imagens e matrizes auxiliares.
"""
from tipos import Image
import numpy as np
import cv2


def imgread(arquivo: str) -> Image:
    """
    Lê e decodifica uma imagem a partir de um buffer IO
    """
    with open(arquivo, mode='rb') as filebuf:
        buf = np.frombuffer(filebuf.read(), dtype=np.uint8)

    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        msg = f'não foi possível parsear "{arquivo}" como imagem'
        raise Exception(msg)

    return img


def imgwrite(img: Image, arquivo: str) -> None:
    """
    Escreve uma matriz como imagem PNG em um arquivo
    """
    if not cv2.imwrite(arquivo, img):
        raise Exception("")


def imgshow(img: Image, nome: str="") -> None:
    """
    Apresenta a imagem em uma janela com um nome
    """
    try:
        cv2.imshow(nome, img)
        cv2.waitKey()
    except KeyboardInterrupt:
        pass
