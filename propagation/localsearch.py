import time
from abc import ABC, abstractmethod
from collections import defaultdict
from heapq import heapify, nsmallest
from itertools import product

import numpy as np

from propagation.optimize import combine_results, Fun, ObjectiveFun, WithCallback, WithHistory
from propagation.searchspace import SearchSpace, Point


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


class SingleLocalSearch(LocalSearch):
    def __init__(self, f, domain, objective, initial=None, seed=None):
        super(f, domain, objective, initial, seed)
        self.current = None

    def next(self, results):
        m = min((self.objective(r), point) for r, point in results)
        self.current = m[1]
        return m

    def best(self):
        return self.current


class PopulationLocalSearch(LocalSearch):
    def __init__(self, f, domain, objective, initial=None, seed=None):
        super(f, domain, objective, initial, seed)
        self.points = defaultdict(list)
        self.solutions = []

    def next(self, results):
        self.register(results)
        return self.solutions[0]

    def explore_point(self, point, force=False):
        if (self.dom.in_bounds(point) and self.visited(point)) or force:
            super(point)
            return True
        return False

    def visited(self, point):
        return point in self.points

    def register(self, results):
        for result, point in results:
            self.points[point].append(tuple(result))
        self.solutions = [
            (self.objective(combine_results(r)), point) for point, r in self.points.items() if r
        ]
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


class SingleHillclimb(SingleLocalSearch):
    def next(self, results):
        super(results)
        self.explore_all_dirs(self.best())


class SingleStochasticHillclimb(SingleHillclimb):
    def next(self, results):
        super(results)


def iterate_stochastic(self, steps=1, k_best=1, n_dirs=1):
    for point in self.best(k_best):
        for _ in range(n_dirs):
            self.explore_random_dir(point, steps)


def iterate_steep(self, steps=1, k_best=1):
    for point in self.best(k_best):
        self.explore_all_dirs(point, steps)


def hillclimb(sim, num=1, timeout=60, sources=None, samples=1000):
    dom = {
        # 'edge_probability': (0., 0.3, .001),
        'discount_factor': (0.0, 1.0, 0.01),
        'corr': (0.0, 0.005, 0.0001),
    }
    print(f'opt: {dom}')

    opts = optimize_all_features(
        sim, domain=dom, sources=sources, samples=samples, explore_current_point=True, num=num
    )

    # print(list(opts.values())[0][0].dom.size())

    best = optimize_all_features(
        sim, domain=dom, sources=sources, samples=samples, explore_current_point=False
    )
    t = time.time()
    while True:
        for feature, os in opts.items():
            for o in os:
                o.register_results()
                for i in range(50):
                    if not o.stuck(steps=i, k_best=2):
                        o.iterate_steep(steps=1, k_best=2)
                        break
                else:
                    best[feature].register_from(o, k_best=2)
                    o.random_restart(n=1, keep=0)

        if time.time() - t > timeout:
            break

    for feature, os in opts.items():
        for o in os:
            best[feature].register_from(o, k_best=2)

    for feature, o in best.items():
        set_params(o.best(), sim, feature)

    return {
        feature: [b.state()] + [o.state() for o in opts[feature]] for feature, b in best.items()
    }


def stochastic_hillclimb(sim, num=1, timeout=60, sources=None, samples=1000):
    dom = {
        # 'edge_probability': (0., 0.3, .001),
        'discount_factor': (0.0, 1.0, 0.01),
        'corr': (0.0, 0.005, 0.0001),
    }
    print(f'opt: {dom}')

    random_starts = optimize_all_features(
        sim, domain=dom, sources=sources, samples=samples, explore_current_point=False, num=num
    )

    best = optimize_all_features(
        sim, domain=dom, sources=sources, samples=samples, explore_current_point=True
    )
    t = time.time()
    while True:
        for feature, os in random_starts.items():
            for o in os:
                o.register_results()
                if o.stuck(steps=1, k_best=2):
                    best[feature].register_from(o, k_best=2)
                    o.random_restart(n=1, keep=0)
                o.iterate_stochastic(steps=1, k_best=2)

        if time.time() - t > timeout:
            print('timeout')
            break

    for feature, os in random_starts.items():
        best[feature].register_results()
        for o in os:
            best[feature].register_from(o, k_best=2)

    for feature, o in best.items():
        set_params(o.best(), sim, feature)

    return {
        feature: [b.state()] + [o.state() for o in random_starts[feature]]
        for feature, b in best.items()
    }


def optimize(sim, sources=None, samples=500):
    dom = {
        # 'edge_probability': (0., 0.3, .1),
        'discount_factor': (0.0, 0.1, 0.1),
        'corr': (0.0, 0.1, 0.1),
    }
    print(f'grid: {dom}')
    grid = optimize_all_features(
        sim, domain=dom, sources=sources, samples=samples, explore_current_point=False
    )
    dom = {
        # 'edge_probability': (0., 0.3, .001),
        'discount_factor': (0.0, 1.0, 0.01),
        'corr': (0.0, 0.005, 0.0001),
    }
    print(f'opt: {dom}')
    opts = optimize_all_features(sim, domain=dom, sources=sources, samples=samples)

    for o in grid.values():
        o.explore_full_grid()
        # o.full_grid(force=True) # Repeat

    for o in opts.values():
        o.explore_random_point(10)

    for feature, o in opts.items():
        grid[feature].register_results()
        o.register_from(grid[feature])
    print('grid done', flush=True)

    for _ in range(20):
        for o in opts.values():
            o.iterate_stochastic(steps=5, k_best=10)
            # o.reexplore_best(10)
    print('stochastic done', flush=True)

    for _ in range(20):
        for o in opts.values():
            o.iterate_steep(k_best=5)
            # o.reexplore_best(5)
    print('hillclimb done', flush=True)

    for feature, o in opts.items():
        set_params(o.best(), sim, feature)

    return {feature: o.state() for feature, o in opts.items()}


# sourcery skip: hoist-if-from-if, merge-nested-ifs, remove-redundant-if
if __name__ == "__main__":
    from .simulation import Simulation

    sim = Simulation.from_files(
        'data/anon_graph_inner_neos_20201110.npz', 'data/sim_features_neos_20201110.csv', seed=3
    )
    # with mpi.futures(sim) as sim:
    if True:
        if sim is not None:
            hillclimb(sim, sources=2, samples=10, num=2)
            print(sim.params.to_csv())
