from optimization.findroot import Fun, ObjectiveFun
from optimization.searchspace import SearchSpace, Point, middle_value, bisect


class MonotoneRoot:
    def __init__(
        self, f: Fun, domain: SearchSpace, objective: ObjectiveFun, initial=None, seed=None
    ):  # implement initial
        assert len(domain.dims) == 1
        self.dim = tuple(domain.dims)[0]
        self.bound = domain.bounds[self.dim]
        self.f = f
        self.objective = objective

    def __iter__(self):
        def _middle():
            return Point({self.dim: middle_value(self.bound)})

        self.mid = _middle()
        self.result = self.f(self.mid)

        while True:
            result = self.objective(self.result)
            lower, upper = bisect(middle_value(self.bound), self.bound)
            mid = _middle()
            print(f'f({mid})={result} (={self.result}) {self.bound}')

            if not (lower and upper):  # could be improved to use midpoint of remaining
                return result, mid
            print(result)
            self.bound = upper if result < 0 else lower
            self.mid = _middle()
            self.result = self.f(self.mid)
            yield result, mid

    def state(self):
        return self.result, self.bound

    def best(self):
        return self.mid
