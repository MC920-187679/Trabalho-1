from argparse import ArgumentParser, ArgumentTypeError
from tipos import Image, Kernel, Argumentos

import numpy as np
import sys

from inout import imgread, imgwrite, imgshow
from filtro import FILTRO
from lib import (
    scipy_convolve, opencv_convolve,
    combina, transforma_limites, trunca,
    Borda, Backend, Limitador
)


def convolucao(img: Image, args: Argumentos) -> Image:
    """
    Aplicação e combinação das convoluções.

    Parâmetros
    ----------
    img: np.ndarray
        Matriz representando a imagem lida.
    args: Argumentos-like object
        Objeto com os argumentos da linha de comando.

    Retorno
    -------
    img: np.ndarray
        Imagem processada.
    """

    # aplica um kernel e retorna a matriz com
    # os floats resultantes
    def aplica(kernel: Kernel) -> np.ndarray:
        # convolução com o backend e bordas selecionados
        res = args.backend(img, kernel, args.borda)

        # tratamento antes da combinação, quando pedido
        if args.antes:
            res = args.limitador(res).astype(float)

        return res

    # lista com os resultados de cada kernel
    resultado = [aplica(kernel) for kernel in args.kernels]
    # combina os resultado e limita para 256 níveis
    img = args.limitador(combina(*resultado))
    return img


# # # # # # # # # # # # # # #
# Tratamento dos argumentos #

def kernel(nome: str) -> Kernel:
    """
    Recuperação dos kernels pré-definidos.
    """
    try:
        return FILTRO[nome]
    except KeyError:
        msg = f'kernel de convolução inválido: {nome}'
        raise ArgumentTypeError(msg)

def borda(nome: str) -> Borda:
    """
    Processamento dos argumentos de tratamento de bordas.
    """
    try:
        return Borda[nome]
    except KeyError:
        msg = f'opção de tratamento de borda inválida: {nome}'
        raise ArgumentTypeError(msg)


# parser de argumentos
description = 'Ferramenta simples de aplicação dos filtros de convolução do Trabalho 1.'
usage = '%(prog)s [OPTIONS] INPUT KERNEL [KERNEL ...]'

parser = ArgumentParser(description=description, usage=usage, allow_abbrev=False)
# argumentos necessários
parser.add_argument('input', metavar='INPUT', type=str,
                    help='imagem de entrada')
parser.add_argument('kernels', type=kernel, metavar='KERNEL', nargs='+',
                    help='filtros a serem aplicados na imagem')
# opções de saída
parser.add_argument('-o', '--output', type=str, action='append', metavar='FILE',
                    help='arquivo para gravar o resultado')
parser.add_argument('-f', '--force-show', action='store_true',
                    help='sempre mostra o resultado final em uma janela')
# configurações da convolução
parser.add_argument('-a', action='store_true',
                    help='aplica a transformação dos níveis antes de combinar as imagens')
parser.add_argument('-b', '--borda',
                    type=borda, choices=Borda, default=Borda.reflexao_pula_um,
                    help='muda o tratamento de borda para a opção dada (PADRÃO: reflexao_pula_um)')
parser.add_argument('-t', '--transflin', dest='limitador', action='store_const',
                    const=transforma_limites, default=trunca,
                    help='transforma linearmente os resultados do filtro para 256 níveis (PADRÃO: trunca o resultado)')
parser.add_argument('--scipy', dest='backend', action='store_const',
                    default=scipy_convolve, const=scipy_convolve,
                    help='faz a concolução com a biblioteca OpenCV')
parser.add_argument('--opencv', dest='backend', action='store_const',
                    default=scipy_convolve, const=opencv_convolve,
                    help='faz a concolução com a biblioteca SciPy (PADRÃO)')


if __name__ == "__main__":
    args = parser.parse_intermixed_args()

    # entrada
    arquivo = args.input
    img = imgread(arquivo)

    # convoluções
    img = convolucao(img, args)

    # saída
    if args.output:
        for output in args.output:
            try:
                imgwrite(img, output)
            # em caso de erro, mostra o erro
            # mas continua a execução
            except ValueError as err:
                print(err, file=sys.stderr)

    if args.output is None or args.force_show:
        imgshow(img, arquivo)
