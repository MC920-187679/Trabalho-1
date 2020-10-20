from argparse import ArgumentParser, ArgumentTypeError
from inout import imgread, imgwrite, imgshow
from tipos import Image, Kernel, Argumentos
from filtro import FILTRO
from typing import Callable

from lib import (
    scipy_convolve, opencv_convolve,
    combina, transforma_limites, trunca,
    Borda, Backend, Limitador
)


def convolucao(img: Image, args: Argumentos) -> Image:
    def aplica(kernel):
        res = args.backend(img, kernel, args.borda)

        if args.antes:
            res = args.limitador(res).astype(float)

        return res

    resultado = [aplica(kernel) for kernel in args.kernels]
    img = args.limitador(combina(*resultado))
    return img


def kernel(nome: str) -> Kernel:
    try:
        return FILTRO[nome]
    except KeyError:
        msg = f'kernel de convolução inválido: {nome}'
        raise ArgumentTypeError(msg)

def borda(nome: str) -> Borda:
    try:
        return Borda[nome]
    except KeyError:
        msg = f'opção de tratamento de borda inválida: {nome}'
        raise ArgumentTypeError(msg)


# parser de argumentos
description = 'Ferramenta de processamentos simples de imagem.'

parser = ArgumentParser(description=description, allow_abbrev=False)
parser.add_argument('input', metavar='INPUT', type=str,
                    help='imagem de entrada')
parser.add_argument('-a', '--antes', action='store_true')
parser.add_argument('-b', '--borda', type=borda, choices=Borda, default=Borda.reflexao,
                    help='muda o tratamento de borda para a opção dada')
parser.add_argument('-t', '--transf', action='store_const', const=transforma_limites, default=trunca, dest='limitador')
parser.add_argument('-s', '--scipy', action='store_const', const=scipy_convolve, default=scipy_convolve, dest='backend')
parser.add_argument('-c', '--opencv', action='store_const', const=opencv_convolve, default=scipy_convolve, dest='backend')
parser.add_argument('-f', '--force-show', action='store_true',
                    help='sempre mostra o resultado final em uma janela')
parser.add_argument('-o', '--output', type=str, action='append', metavar='FILE',
                    help='arquivo para gravar o resultado')
parser.add_argument('kernels', type=kernel, metavar='KERNEL', nargs='+',
                    help='operações que devem ser feitas na imagem')

if __name__ == "__main__":
    args: Argumentos = parser.parse_intermixed_args()
    # print(args)

    # entrada
    arquivo = args.input
    img = imgread(arquivo)

    # operações
    img = convolucao(img, args)

    # saída
    if args.output:
        for output in args.output:
            imgwrite(img, output)

    if args.output is None or args.force_show:
        imgshow(img, arquivo)
