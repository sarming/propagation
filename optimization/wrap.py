import time
from abc import ABC

from optimization.findroot import FindRoot, abs_value, FindRootFactory


class Wrapper(ABC):
    @classmethod
    def wrap(cls, wrapee: FindRootFactory, *args, **kwargs):
        return lambda *iargs, **ikwargs: cls(wrapee(*iargs, **ikwargs), *args, **kwargs)


class WithHistory(Wrapper):
    def __init__(self, o: FindRoot):
        self.o = o
        self.history = []

    def __iter__(self):
        for res in self.o:
            self.history.append(res)
            yield res

    def best(self):
        return min(self.history, key=abs_value)[1]

    def state(self):
        return self.o.state(), self.history


class WithAllTimeBest(Wrapper):
    def __init__(self, o: FindRoot):
        self.o = o
        self.current_best = (float("inf"), None)

    def __iter__(self):
        for res in self.o:
            if res[0] <= self.current_best[0]:
                self.current_best = res
            yield res

    def best(self):
        return self.current_best[1]

    def state(self):
        return self.o.state(), self.current_best


class WithCallback(Wrapper):
    def __init__(self, o: FindRoot, callback, *vars):
        self.o = o
        self.callback = callback
        self.vars = vars

    def __iter__(self):
        for res in self.o:
            if self.callback(res, self.o, *self.vars):
                continue
            yield res

    def state(self):
        return self.o.state()


class WithTimeout(Wrapper):
    def __init__(self, o: FindRoot, timeout_seconds):
        self.o = o
        self.timeout = timeout_seconds

    def __iter__(self):
        t = time.time()
        for res in self.o:
            if t + self.timeout >= time.time():
                return res
            yield res

    def state(self):
        return self.o.state()


class WithPrint(Wrapper):
    def __init__(self, o: FindRoot, *info, print_state=False, flush=False):
        self.o = o
        self.info = info
        self.print_state = print_state
        self.flush = flush

    def __iter__(self):
        for res in self.o:
            if self.info:
                print(*self.info, end=": ")
            print(res, end=" ")
            if self.print_state:
                print(f"<<{self.o.state()}>>", end=" ")
            print(flush=self.flush)
            yield res

    def state(self):
        return self.o.state()


class FindRootParallel:
    def __init__(self, opts):
        self.opts = opts
        self.active = [iter(o) for o in opts]

    def __iter__(self):
        if not self.active:
            self.active = [iter(o) for o in self.opts]
        return self

    def _next_or_remove(self, o):
        try:
            return [o.__next__()]
        except StopIteration:
            self.active.remove(o)
            return []

    def __next__(self):
        current_active = self.active.copy()
        res = sum((self._next_or_remove(o) for o in current_active), [])
        if not self.active:
            raise StopIteration()
        return min(res, key=abs_value)

    def state(self):
        return [o.state() for o in self.opts]


class DictFindRoot:
    def __init__(self, opts):
        self.opts = opts
        self.active = {key: iter(o) for key, o in opts.items()}

    def __iter__(self):
        if not self.active:
            self.active = {key: iter(o) for key, o in self.opts.items()}
        return self

    def _next_or_remove(self, key):
        try:
            return [(key, self.active[key].__next__())]
        except StopIteration:
            del self.active[key]
            return []

    def __next__(self):
        current_active = self.active.copy()
        res = dict(sum((self._next_or_remove(key) for key in current_active), []))
        if not self.active:
            raise StopIteration()
        return res

    def best(self):
        return {key: opt.best() for key, opt in self.opts.items()}

    def state(self):
        return {key: opt.state() for key, opt in self.opts.items()}
