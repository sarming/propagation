from functools import wraps
from typing import Protocol, Iterator, Tuple, Any, Callable, TypeVar, Optional

from optimization.searchspace import SearchSpace, Point

Result = TypeVar('T')
Fun = Callable[[Point], Result]
ObjectiveFun = Callable[[Result], float]


class FindRoot(Protocol):
    def __iter__(self) -> Iterator[Tuple[float, Point]]:
        ...

    def state(self) -> Any:
        ...


class FindRootFactory(Protocol):
    def __call__(
        self,
        f: Fun,
        domain: SearchSpace,
        objective: ObjectiveFun,
        initial: Optional[Point],
        seed=None,
    ) -> FindRoot:
        ...


def abs_value(res: Tuple[float, Point]) -> float:
    return abs(res[0])


def logging(fun):
    """Simple function decorator that prints every function invocation."""

    @wraps(fun)
    def wrapper(*args, **kwargs):
        print(f'calling {fun.__name__}({args},{kwargs})')
        return fun(*args, **kwargs)

    return wrapper
