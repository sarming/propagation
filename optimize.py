from collections import defaultdict
from functools import wraps
from heapq import heappush, nsmallest
from itertools import product
from math import floor, prod

import numpy as np


def invert_monotone(fun, goal, lb, ub, eps, logging=False):
    """Find fun^-1( goal ) by binary search.

    Note:
        For correctness fun has to be monotone up to eps, viz. fun(x) <= fun(x+eps) for all x
        This is an issue with stochastic functions.

    Args:
        fun: Monotone Function.
        goal: Goal value.
        lb: Initial lower bound.
        ub: Initial upper bound.
        eps: Precision at which to stop.
        logging: Print function calls and results.

    Returns: x s.t. lb<x<ub and there is y with |x-y|<=eps and fun(y)=goal
    """
    mid = (ub + lb) / 2
    if ub - lb < eps:
        return mid
    if logging:
        print(f'f({mid})=', end='')
    f = fun(mid)
    if logging:
        print(f"{f}{'<' if f < goal else '>'}{goal} [{lb},{ub}]")
    if f < goal:
        return invert_monotone(fun, goal, mid, ub, eps, logging)
    else:
        return invert_monotone(fun, goal, lb, mid, eps, logging)


def logging(fun):
    """Simple function decorator that prints every function invocation."""

    @wraps(fun)
    def wrapper(*args, **kwargs):
        print(f'calling {fun.__name__}({args},{kwargs})')
        return fun(*args, **kwargs)

    return wrapper


def single_objective(sim, feature, statistic, absolute=True):
    assert statistic in {'mean_retweets', 'retweet_probability'}
    goal = sim.stats.at[feature, statistic]

    def obj(result):
        mean_retweets, retweet_probability = result  # list(result)
        result = mean_retweets if statistic == 'mean_retweets' else retweet_probability
        if absolute:
            return abs(result - goal)
        return result - goal

    return obj


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
    if not isinstance(bound, tuple):  # only continuous bounds need to be discretized
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


class Domain:
    def __init__(self, bounds):
        self.bounds = bounds
        self.dims = bounds.keys()

    class Point(dict):  # https://stackoverflow.com/a/1151686/153408
        def __key(self):
            return tuple((k, self[k]) for k in sorted(self))

        def __hash__(self):
            return hash(self.__key())

        def __eq__(self, other):
            return self.__key() == other.__key()

        def __lt__(self, other):
            return self.__key() < other.__key()

    def to_point(self, dictlike):
        return Domain.Point({dim: dictlike[dim] for dim in self.dims})  # Use only keys from self.dims

    def in_bounds(self, point):
        for dim in self.dims:
            if not in_bound(point[dim], self.bounds[dim]):
                return False
        return True

    def step(self, point, dim, steps, neg=False):
        if isinstance(steps, dict):
            steps = steps[dim]
        if neg:
            steps = -steps
        point = Domain.Point(point)  # Copy point
        bound = self.bounds[dim]
        old = point[dim]
        if isinstance(bound, tuple):
            lb, ub, stepwidth = bound
            point[dim] = discretize(old, bound) + steps * stepwidth
            if point[dim] < lb:
                point[dim] = lb
            if point[dim] > ub:
                point[dim] = ub
        else:
            point[dim] = bound[(bound.index(old) + steps) % len(bound)]
        return point

    def random_point(self, n=None, rng=np.random.default_rng()):
        def rnd_point():
            return Domain.Point({dim: random_value(self.bounds[dim], rng) for dim in self.dims})

        if n is None:
            return rnd_point()
        return [rnd_point() for _ in range(n)]

    def all_points(self):
        values = product(*[all_values(bound) for bound in self.bounds.values()])
        for v in values:
            yield dict(zip(self.bounds.keys(), v))

    def size(self):
        return prod(bound_size(bound) for bound in self.bounds.values())


class Optimize:
    def __init__(self, f, domain, objective, seed=None):
        self.f = f
        self.dom = domain
        self.objective = objective
        self.rng = np.random.default_rng(seed)

        self.dirs = list(product(self.dom.dims, [False, True]))

        self.points = defaultdict(list)
        self.solutions = []
        self.raw_results = []

    @classmethod
    def all_features(cls, sim, domain=None, statistic='mean_retweets', sources=None, samples=500,
                     add_current_params=True):
        return {feature: cls.feature(sim, feature, domain=domain, statistic=statistic, sources=sources, samples=samples,
                                     add_current_params=add_current_params)
                for feature in sim.features}

    @classmethod
    def feature(cls, sim, feature, domain=None, statistic='mean_retweets', sources=None, samples=500,
                add_current_params=True):
        if domain is None:
            domain = {'edge_probability': (0., 0.3, .001),
                      'at_least_one': [True, False],
                      'discount_factor': (0., 1., .1),
                      'corr': (0., 1., .001),
                      'max_nodes': range(100, 860, 20),
                      'max_depth': [100],
                      }
        if not isinstance(domain, Domain):
            domain = Domain(domain)
        objective = single_objective(sim, feature, statistic)
        f = lambda point: sim.simulate(feature, point, sources, samples, True)
        self = cls(f, domain, objective, sim.seed.spawn(1)[0])
        if add_current_params:
            self.add_point(sim.params.astype('object').loc[feature])
        return self

    def add_point(self, point):
        if not isinstance(point, Domain.Point):
            point = self.dom.to_point(point)
        if not self.dom.in_bounds(point) or self.visited(point):
            return False
        result = self.f(point)
        self.raw_results.append((result, point))
        self.points[point]  # Access to add key to dict
        # print(f'add point {point}')
        return True

    def evaluate(self):
        new_solutions = [(self.objective(result), point) for result, point in self.raw_results]
        self.add_solutions(new_solutions)
        if new_solutions:
            # print(f'current best:{self.solutions[0]}')
            print('.', flush=True, end='')
        self.raw_results = []

    def add_solutions(self, solutions):
        for s in solutions:
            value, point = s
            self.points[point].append(value)
            heappush(self.solutions, s)

    def visited(self, point):
        return point in self.points

    def take_all_dirs(self, point, steps=1):
        for dim, neg in self.dirs:
            self.add_point(self.dom.step(point, dim, steps, neg))

    def take_random_dir(self, point, steps=1):
        dirs = self.rng.permutation(self.dirs)
        for dim, neg in dirs:
            p = self.dom.step(point, dim, steps, neg)
            if self.add_point(p):
                return True
        return False

    def iterate_stochastic(self, steps=1, k_best=1, n_dirs=1):
        self.evaluate()
        for point in self.best(k_best):
            for i in range(n_dirs):
                self.take_random_dir(point, steps)

    def iterate_steep(self, steps=1, k_best=1):
        self.evaluate()
        for point in self.best(k_best):
            self.take_all_dirs(point, steps)

    def full_grid(self):
        # self.evaluate() # Can ignore old results
        for point in self.dom.all_points():
            self.add_point(point)

    def best(self, n=None):
        if n is None:
            return self.solutions[0][1]
        return [x[1] for x in self._best_solutions(n)]

    def _best_solutions(self, n=1):
        if n == 1 and self.solutions:
            return [self.solutions[0]]
        return list(nsmallest(n, self.solutions))

    def add_random_point(self, n=1):
        for point in self.dom.random_point(n, self.rng):
            self.add_point(point)

    def random_restart(self, n=1, keep=1):
        self.evaluate()
        old_solutions = self._best_solutions(keep)
        self.solutions = []
        self.points = defaultdict(list)
        self.add_random_point(n)
        return old_solutions

    def set_best(self, sim, feature):
        for dim, value in self.solutions[0][1].items():
            sim.params.at[feature, dim] = value


def optimize(sim, sources=None, samples=500):
    dom = {
        # 'edge_probability': (0., 0.3, .1),
        'discount_factor': (0., 1., .1),
        'corr': (0., 1., .1)
    }
    print(f'grid: {dom}')
    grid = Optimize.all_features(sim, sources=sources, samples=samples, domain=dom)
    dom = {
        # 'edge_probability': (0., 0.3, .001),
        'discount_factor': (0., 1., .01),
        'corr': (0., .005, .0001)
    }
    print(f'opt: {dom}')
    opts = Optimize.all_features(sim, domain=dom, sources=sources, samples=samples, add_current_params=False)

    for o in grid.values():
        o.full_grid()

    for o in opts.values():
        o.add_random_point(1)

    for feature, o in opts.items():
        grid[feature].evaluate()
        o.add_solutions(grid[feature].solutions)
    print('grid done', flush=True)

    for o in opts.values():
        for _ in range(20):
            o.iterate_stochastic(steps=5, k_best=10)
    print('stochastic done', flush=True)

    for o in opts.values():
        for _ in range(20):
            o.iterate_steep(k_best=5)
    print('hillclimb done', flush=True)

    for feature, o in opts.items():
        o.set_best(sim, feature)

    return {feature: o.solutions for feature, o in opts.items()}


if __name__ == "__main__":
    from simulation import Simulation

    sim = Simulation.from_files('data/anon_graph_inner_neos_20201110.npz', 'data/sim_features_neos_20201110.csv',
                                seed=3)
    # with mpi.futures(sim) as sim:
    if True:
        if sim is not None:
            optimize(sim, 1, 1)
            print(sim.params.to_csv())
