from typing import Callable

from .findroot import FindRoot, abs_value


class FindRootParallel:
    @classmethod
    def wrap(cls, wrapped_cls: Callable[..., FindRoot], num=1):
        return lambda *vars, **kwargs: cls([wrapped_cls(*vars, **kwargs) for _ in range(num)])

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
