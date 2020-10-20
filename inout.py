"""
Funções de IO com as imagens.
"""
from tipos import Image
import numpy as np
import cv2


def imgread(arquivo: str) -> Image:
    """
    Lê um arquivo de imagem.
    """
    # abre o arquivo fora do OpenCV, para que o
    # Python trate os erros de IO
    with open(arquivo, mode='rb') as filebuf:
        buf = np.frombuffer(filebuf.read(), dtype=np.uint8)

    # só resta tratar problemas de decodificação
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        msg = f'não foi possível parsear "{arquivo}" como imagem'
        raise ValueError(msg)

    return img


def imgwrite(img: Image, arquivo: str) -> None:
    """
    Escreve uma matriz como imagem PNG em um arquivo.
    """
    # indica para o caller quando a imagem NÃO for salva
    if not cv2.imwrite(arquivo, img):
        msg = f'não foi possível salvar a imagem em "{arquivo}"'
        raise ValueError(msg)


def imgshow(img: Image, nome: str="") -> None:
    """
    Apresenta a imagem em uma janela com um nome
    """
    try:
        cv2.imshow(nome, img)
        cv2.waitKey()
    # Ctrl-C não são erros nesse caso
    except KeyboardInterrupt:
        pass
