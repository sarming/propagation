import time
from abc import ABC
from typing import Callable

from .findroot import FindRoot, abs_value


class Wrapper(ABC):
    @classmethod
    def wrap(cls, wrapped_cls: Callable[..., FindRoot], *args, **kwargs):
        return lambda *iargs, **ikwargs: cls(wrapped_cls(*iargs, **ikwargs), *args, **kwargs)

    def __init__(self, wrapped: FindRoot) -> None:
        self.wrapped = wrapped

    def __getattr__(self, name):
        if hasattr(self.wrapped, name):
            return getattr(self.wrapped, name)
        raise AttributeError


class WithHistory(Wrapper):
    def __init__(self, o: FindRoot):
        super().__init__(o)
        self.wrapped = o
        self.history = []

    def __iter__(self):
        for res in self.wrapped:
            self.history.append(res)
            yield res

    def best(self):
        return min(self.history, key=abs_value)[1]

    def state(self):
        return self.wrapped.state(), self.history


class WithAllTimeBest(Wrapper):
    def __init__(self, o: FindRoot):
        super().__init__(o)
        self.current_best = (float("inf"), None)

    def __iter__(self):
        for res in self.wrapped:
            if res[0] <= self.current_best[0]:
                self.current_best = res
            yield res

    def best(self):
        return self.current_best[1]

    def state(self):
        return self.wrapped.state(), self.current_best


class WithCallback(Wrapper):
    @classmethod
    def wrap(
        cls,
        wrapped_cls: Callable[..., FindRoot],
        callback,
        *args,
        wrapped_args=[],
        wrapped_kwargs=[],
    ):
        return lambda *iargs, **ikwargs: cls(
            wrapped_cls(*iargs, **ikwargs),
            callback,
            *args,
            *[iargs[i] for i in wrapped_args],
            **{k: ikwargs[k] for k in wrapped_kwargs},
        )

    def __init__(self, o: FindRoot, callback, *args, **kwargs):
        super().__init__(o)
        self.callback = callback
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        for res in self.wrapped:
            if self.callback(res, self.wrapped, *self.args, **self.kwargs):
                continue  # suppress res
            yield res

    def state(self):
        return self.wrapped.state()


class WithTimeout(Wrapper):
    def __init__(self, o: FindRoot, timeout_seconds):
        super().__init__(o)
        self.timeout = timeout_seconds

    def __iter__(self):
        t = time.time()
        for res in self.wrapped:
            yield res
            if t + self.timeout <= time.time():
                return

    def state(self):
        return self.wrapped.state()


class WithPrint(Wrapper):
    def __init__(self, o: FindRoot, *info, print_state=False, flush=False):
        super().__init__(o)
        self.info = info
        self.print_state = print_state
        self.flush = flush

    def __iter__(self):
        for res in self.wrapped:
            if self.info:
                print(*self.info, end=": ")
            print(res, end=" ")
            if self.print_state:
                print(f"<<{self.wrapped.state()}>>", end=" ")
            print(flush=self.flush)
            yield res

    def state(self):
        return self.wrapped.state()
