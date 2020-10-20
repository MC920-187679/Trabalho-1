from argparse import ArgumentParser
from inout import imgread, imgwrite, imgshow
from tipos import Image, Kernel
from typing import Callable

from lib import (
    scipy_convolve, opencv_convolve,
    combina, transforma_limites, trunca,
    Borda, Backend, Limitador
)


def convolucao(backend: Backend, img: Image, borda: Borda, limita: Limitador, antes: bool, *kernels: Kernel) -> Image:
    resultado = []
    for kernel in kernels:
        res = backend(img, kernel, borda)
        if antes:
            res = limita(res).astype(float)
        resultado.append(res)

    if len(kernels) < 2:
        ...

    img = limita(combina(*resultado))
    return img


def borda(nome: str) -> Borda:
    return Borda[nome.upper()]

# parser de argumentos
description = 'Ferramenta de processamentos simples de imagem.'

parser = ArgumentParser(description=description, allow_abbrev=False)
parser.add_argument('input', type=str, metavar='INPUT',
                    help='imagem de entrada')
parser.add_argument('-a', '--antes', action='store_true')
parser.add_argument('-b', '--borda', type=Borda.argtype, choices=Borda, default=Borda.reflexao)
parser.add_argument('-t', '--transf', action='store_const', const=transforma_limites, default=trunca, dest='limitador')
parser.add_argument('-s', '--scipy', action='store_const', const=scipy_convolve, default=scipy_convolve, dest='backend')
parser.add_argument('-c', '--opencv', action='store_const', const=opencv_convolve, default=scipy_convolve, dest='backend')
parser.add_argument('-f', '--force-show', action='store_true',
                    help='sempre mostra o resultado final em uma janela')
parser.add_argument('-o', '--output', type=str, action='append', metavar='FILE',
                    help='arquivo para gravar o resultado')
parser.add_argument('ops', type=str, metavar='OPERATION', nargs='+',
                    help='operações que devem ser feitas na imagem')

if __name__ == "__main__":
    args = parser.parse_intermixed_args()
    print(args)

    # # entrada
    # arquivo = args.input
    # img = imgread(arquivo)

    # # operações
    # # img = aplica_ops(img, args.ops)

    # # saída
    # if args.output:
    #     for output in args.output:
    #         imgwrite(img, output)

    # if args.output is None or args.force_show:
    #     imgshow(img, arquivo)
