from abc import ABC, abstractmethod
from collections import defaultdict
from heapq import heapify, nsmallest
from itertools import product

import numpy as np

from optimization.findroot import (
    Fun,
    ObjectiveFun,
)
from optimization.searchspace import SearchSpace, Point
from optimization.wrap import WithHistory, WithCallback


class LocalSearch(ABC):
    def __init__(
        self, f: Fun, domain: SearchSpace, objective: ObjectiveFun, initial: Point = None, seed=None
    ):
        self.f = f
        self.dom = domain
        self.objective = objective
        self.rng = np.random.default_rng(seed)
        self.dirs = list(product(self.dom.dims, [False, True]))
        self.raw_results = []
        if initial is not None:
            self.explore_point(initial)

    def explore_point(self, point, force=False):
        if not isinstance(point, Point):
            point = self.dom.to_point(point)
        result = self.f(point)
        self.raw_results.append((result, point))
        # print(f'add point {point}')

    def explore_all_dirs(self, point, steps=1):
        for dim, neg in self.dirs:
            self.explore_point(self.dom.step(point, dim, steps, neg))

    def explore_random_dir(self, point, steps=1):
        dirs = self.rng.permutation(self.dirs)
        for dim, neg in dirs:
            p = self.dom.step(point, dim, steps, neg)
            if self.explore_point(p):
                return True
        return False

    def explore_full_grid(self, force=False):
        # self.evaluate() # Can ignore old results
        for point in self.dom.all_points():
            self.explore_point(point, force)

    def explore_random_point(self, n=1):
        for point in self.dom.random_point(n, self.rng):
            self.explore_point(point)

    def __iter__(self):
        return self

    def __next__(self):
        new_results = self.raw_results
        self.raw_results = []
        return self.next(new_results)

    @classmethod
    def with_callback(cls, callback, *vars):
        def init(f: Fun, domain, objective, initial=None, seed=None):
            o = cls(f, domain, objective, initial, seed)
            return WithHistory(WithCallback(o, callback, *vars))

        return init

    @abstractmethod
    def next(self, new_results):
        ...

    @abstractmethod
    def state(self):
        ...


class SingleLocalSearch(LocalSearch):
    def __init__(self, f, domain, objective, initial=None, seed=None):
        super().__init__(f, domain, objective, initial, seed)
        self.current = None

    def next(self, results):
        m = min((self.objective(r), point) for r, point in results)
        self.current = m
        return m

    def best(self):
        return self.current[1]

    def state(self):
        return self.current


class PopulationLocalSearch(LocalSearch):
    def __init__(self, f, domain, objective, initial=None, seed=None):
        super().__init__(f, domain, objective, initial, seed)
        self.points = defaultdict(list)
        self.solutions = []
        self.combine_results = None

    def next(self, results):
        self.register(results)
        return self.solutions[0]

    def explore_point(self, point, force=False):
        if (self.dom.in_bounds(point) and self.visited(point)) or force:
            super().explore_point(point)
            return True
        return False

    def visited(self, point):
        return point in self.points

    def _combine(self, results):
        if self.combine_results:
            return self.objective(self.combine_results(results))
        else:
            return sum(self.objective(results)) / len(results)

    def register(self, results):
        for result, point in results:
            self.points[point].append(tuple(result))
        self.solutions = [(self._combine(r), point) for point, r in self.points.items() if r]
        heapify(self.solutions)

    def register_from(self, other, k_best=None):
        """Register k_best points with results from other (all points if k_best is None)."""
        points = other.points.keys() if k_best is None else other.best(k_best)
        self.register((r, p) for p in points for r in other.points[p])
        # self.register((combine_results(other.points[p]), p) for p in points)

    def best(self, n=None):
        if n is None:
            return self.solutions[0][1]
        return [x[1] for x in self._best_solutions(n)]

    def _best_solutions(self, n=1):
        if n == 1 and self.solutions:
            return [self.solutions[0]]
        return list(nsmallest(n, self.solutions))

    def reexplore_best(self, n=1):
        for point in self.best(n):
            self.explore_point(point, force=True)

    def random_restart(self, n=1, return_kbest=1):
        self.points = defaultdict(list)
        self.explore_random_point(n)
        old_solutions = self._best_solutions(return_kbest)
        self.solutions = []
        return old_solutions

    def state(self):
        return self.points


class SingleHillclimb(SingleLocalSearch):
    def next(self, results):
        res = super().next(results)  # Call this first to get new self.best()
        self.explore_all_dirs(self.best())
        return res


class SingleStochasticHillclimb(SingleHillclimb):
    def next(self, results):
        super().next(results)


def iterate_stochastic(self, steps=1, k_best=1, n_dirs=1):
    for point in self.best(k_best):
        for _ in range(n_dirs):
            self.explore_random_dir(point, steps)


def iterate_steep(self, steps=1, k_best=1):
    for point in self.best(k_best):
        self.explore_all_dirs(point, steps)
