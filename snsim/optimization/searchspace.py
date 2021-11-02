import functools
from itertools import product
from math import floor

import numpy as np
from frozendict import frozendict

# Backward compatibility for functools.cache
memoize = functools.cache if hasattr(functools, 'cache') else functools.lru_cache(None)


def in_bound(value, bound):
    if isinstance(bound, tuple):
        return bound[0] <= value <= bound[1]
    return value in bound


def discretize(value, bound):
    if not isinstance(bound, tuple):  # only continuous bounds need to be discretized
        return value
    lb, ub, width = bound
    k = floor((value - lb) / width)
    return lb + k * width


def bound_size(bound):
    if not isinstance(bound, tuple):
        return len(bound)
    lb, ub, width = bound
    return floor((ub - lb) / width)


def all_values(bound):
    if isinstance(bound, tuple):
        lb, ub, width = bound
        while lb < ub:
            yield lb
            lb += width
    else:
        yield from bound


def random_value(bound, rng=np.random.default_rng()):
    if isinstance(bound, tuple):
        lb, ub, _ = bound
        if isinstance(lb, int):
            return discretize(rng.integers(lb, ub, endpoint=True), bound)
        if isinstance(lb, float):
            return discretize(rng.uniform(lb, ub), bound)
        assert False
    return rng.choice(bound)


def middle_value(bound):
    if not isinstance(bound, tuple):
        return bound[len(bound) // 2]
    lb, ub, width = bound
    return discretize((ub + lb) / 2, bound)


def bisect(mid, bound):
    mid = discretize(mid, bound)
    if isinstance(bound, tuple):
        lb, ub, width = bound
        lower = lb, mid, width
        upper = mid, ub, width
        if mid - width < lb:
            lower = []
        if mid + width > ub:
            upper = []
    else:
        m = bound.index(mid)
        lower = bound[0:m]
        upper = bound[m:]

    return lower, upper


@functools.total_ordering
class Point(frozendict):
    # Ordering is just convenience to allow comparison of tuples containing Points
    def __lt__(self, other):
        return hash(self) < hash(other)


class SearchSpace:
    def __init__(self, bounds):
        self.bounds = bounds
        self.dims = bounds.keys()

    def to_point(self, dictlike):
        return Point({dim: dictlike[dim] for dim in self.dims})  # Use only keys from self.dims

    def in_bounds(self, point):
        return all(in_bound(point[dim], self.bounds[dim]) for dim in self.dims)

    def step(self, point, dim, steps, neg=False):
        if isinstance(steps, dict):
            steps = steps[dim]
        if neg:
            steps = -steps
        point = Point(point)  # Copy point
        bound = self.bounds[dim]
        old = point[dim]
        if isinstance(bound, tuple):
            lb, ub, stepwidth = bound
            point[dim] = discretize(old, bound) + steps * stepwidth
            if point[dim] < lb:
                point[dim] = lb
            elif point[dim] > ub:
                point[dim] = ub
        else:
            point[dim] = bound[(bound.index(old) + steps) % len(bound)]
        return point

    def random_point(self, n=None, rng=np.random.default_rng()):
        def rnd_point():
            return Point({dim: random_value(self.bounds[dim], rng) for dim in self.dims})

        if n is None:
            return rnd_point()
        return [rnd_point() for _ in range(n)]

    @memoize
    def all_points(self):
        values = product(*[all_values(bound) for bound in self.bounds.values()])
        return [dict(zip(self.bounds.keys(), v)) for v in values]

    def size(self):
        return np.prod([bound_size(bound) for bound in self.bounds.values()])

    def ndims(self):
        return len(1 for bound in self.bounds.values() if bound_size(bound) > 1)

    def __repr__(self) -> str:
        return "SearchSpace(" + repr(self.bounds) + ")"
