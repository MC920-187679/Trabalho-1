"""
Procolos para tipagem estática com ``mypy``.
"""
from __future__ import annotations
from numpy import ndarray, uint8, int64, float64
from typing import (
    TYPE_CHECKING,
    Type, Protocol, Literal, overload,
    Union, Optional, Tuple, List
)

if TYPE_CHECKING:
    from lib import Borda, Backend, Limitador

class Argumentos(Protocol):
    input: str
    antes: bool
    backend: Backend
    borda: Borda
    force_show: bool
    limitador: Limitador
    output: Optional[List[str]]
    kernels: List[Kernel]



class Kernel(ndarray): # type: ignore
    dtype: Union[Type[int64], Type[float64]]
    ndim: Literal[2] = 2
    shape: Tuple[int, int]


class Image(ndarray): # type: ignore
    """
    Matriz que representam imagens em OpenCV e bibliotecas similares.
    """
    dtype: Type[uint8] = uint8
    ndim: Literal[2] = 2
    shape: Tuple[int, int]

    def copy(self) -> Image:
        ...

    @overload
    def __add__(self, other: Union[Image, int]) -> Image: ...
    @overload
    def __add__(self, other: Union[ndarray, float]) -> ndarray: ...
    def __add__(self, other: Union[ndarray, float]) -> ndarray:
        ...

    @overload
    def __sub__(self, other: Union[Image, int]) -> Image: ...
    @overload
    def __sub__(self, other: Union[ndarray, float]) -> ndarray: ...
    def __sub__(self, other: Union[ndarray, float]) -> ndarray:
        ...

    def __neg__(self) -> Image:
        ...

    def __pos__(self) -> Image:
        ...

    def __abs__(self) -> Image:
        ...

    def __invert__(self) -> Image:
        ...

    @overload
    def __mul__(self, other: Union[Image, int]) -> Image: ...
    @overload
    def __mul__(self, other: Union[ndarray, float]) -> ndarray: ...
    def __mul__(self, other: Union[ndarray, float]) -> ndarray:
        ...

    def __matmul__(self, other: ndarray) -> ndarray:
        ...

    @overload
    def __pow__(self, other: Union[Image, int]) -> Image: ...
    @overload
    def __pow__(self, other: Union[ndarray, float]) -> ndarray: ...
    def __pow__(self, other: Union[ndarray, float]) -> ndarray:
        ...

    @overload
    def __floordiv__(self, other: Union[Image, int]) -> Image: ...
    @overload
    def __floordiv__(self, other: ndarray) -> ndarray: ...
    def __floordiv__(self, other: Union[ndarray, int]) -> ndarray:
        ...

    def __truediv__(self, other: Union[ndarray, float]) -> ndarray:
        ...

    @overload
    def __mod__(self, other: Union[Image, int]) -> Image: ...
    @overload
    def __mod__(self, other: Union[ndarray, float]) -> ndarray: ...
    def __mod__(self, other: Union[ndarray, float]) -> ndarray:
        ...

    def __rshift__(self, other: Union[Image, int]) -> Image:
        ...

    def __lshift__(self, other: Union[Image, int]) -> Image:
        ...

    def __and__(self, other: Union[Image, int]) -> Image:
        ...

    def __xor__(self, other: Union[Image, int]) -> Image:
        ...

    def __or__(self, other: Union[Image, int]) -> Image:
        ...
