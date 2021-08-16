from math import floor
import numpy as np
import pandas as pd
from functools import wraps
from itertools import product
from heapq import heappush, nsmallest


def discretize(value, bound):
    lb, ub, width = bound
    k = floor((value - lb) / width)
    return lb + k * width


class hashabledict(dict):  # https://stackoverflow.com/a/1151686/153408
    def __key(self):
        return tuple((k, self[k]) for k in sorted(self))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return self.__key() == other.__key()


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


class Hillclimb:
    def __init__(self, sim, feature, statistic='mean_retweets', bounds=None, sources=None, samples=1,
                 seed=None):
        if bounds is None:
            bounds = {'edge_probability': (0., 0.3, .001),
                      'at_least_one': [True, False],
                      'discount_factor': (0., 1., .1),
                      'corr': (0., 1., .001),
                      'max_nodes': (100, 860, 20),
                      # 'max_depth': (1,100, 1),
                      }
        self.sim = sim
        self.feature = feature
        self.statistic = statistic
        self.bounds = bounds
        self.dims = bounds.keys()
        self.dirs = list(product(self.dims, [False, True]))
        self.sources = sources
        self.samples = samples
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.redo_probability = 0.0

        assert statistic in {'mean_retweets', 'retweet_probability'}
        self.goal = self.sim.stats.at[self.feature, self.statistic]

        self.solutions = []
        self.points = set()

        self.current_results = []

        self.add_point(sim.params.astype('object').loc[feature])

    def simulate(self, point):
        return self.sim.simulate(self.feature, point, self.sources, self.samples, True)

    def objective(self, result, absolute=True):
        mean_retweets, retweet_probability = result  # list(result)
        result = mean_retweets if self.statistic == 'mean_retweets' else retweet_probability
        if absolute:
            return abs(result - self.goal)
        return result - self.goal

    def add_point(self, point):
        if not isinstance(point, hashabledict):
            point = hashabledict({dim: point[dim] for dim in self.dims})  # Use only keys from self.dims
        if not self.visited(point) and self.in_bounds(point):
            # print(f'add point {point}')
            result = self.simulate(point)
            self.current_results.append((result, point))
            self.points.add(point)
            return True
        return False

    def evaluate(self):
        for result, point in self.current_results:
            value = self.objective(result)
            heappush(self.solutions, (value, point))
        if self.solutions:
            print(f'current best:{self.solutions[0]}')
        self.current_results = []

    def in_bounds(self, point):
        def check(bound, value):
            if isinstance(bound, tuple):
                return bound[0] <= value <= bound[1]
            if isinstance(bound, list):
                return value in bound
            assert False

        for dim in self.dims:
            if not check(self.bounds[dim], point[dim]):
                return False
        return True

    def visited(self, point):
        if self.rng.random() <= self.redo_probability:
            print(f'Redoing {point}')
            return False
        return point in self.points

    def take_step(self, point, dim, steps, neg=False):
        if isinstance(steps, dict):
            steps = steps[dim]
        if neg:
            steps = -steps
        point = hashabledict(point)  # Copy point
        bound = self.bounds[dim]
        if isinstance(bound, list):
            point[dim] = bound[(bound.index(point[dim]) + steps) % len(bound)]
        else:
            stepwidth = bound[2]
            point[dim] += steps * stepwidth
        return self.add_point(point)

    def take_all_dirs(self, point, steps):
        for dim in self.dims:
            self.take_step(point, dim, steps)
            self.take_step(point, dim, steps, neg=True)

    def take_random_dir(self, point, steps):
        for _ in range(100):
            dim, neg = self.rng.choice(self.dirs)
            if self.take_step(point, dim, steps, neg):
                return
        print(f'Isolated point? {point}')

    def iterate_stochastic(self, steps, k_best=1, n_dirs=1):
        self.evaluate()
        for point in self.best_points(k_best):
            for i in range(n_dirs):
                self.take_random_dir(point, steps)

    def iterate_steep(self, steps, k_best=1):
        self.evaluate()
        for point in self.best_points(k_best):
            self.take_all_dirs(point, steps)

    def random_points(self, n=1):
        def rnd(bound):
            if isinstance(bound, tuple):
                lb, ub, _ = bound
                if isinstance(lb, int):
                    return discretize(self.rng.integers(lb, ub, endpoint=True), bound)
                if isinstance(lb, float):
                    return discretize(self.rng.uniform(lb, ub), bound)
                assert False
            if isinstance(bound, list):
                return self.rng.choice(bound)
            assert False

        points = [hashabledict({dim: rnd(self.bounds[dim]) for dim in self.dims}) for _ in range(n)]
        return points

    def best_points(self, n=1):
        return [x[1] for x in self._best_solutions(n)]

    def _best_solutions(self, n=1):
        if n == 1 and self.solutions:
            return [self.solutions[0]]
        return list(nsmallest(n, self.solutions))

    def random_restart(self, n=1, keep=1):
        self.evaluate()
        old_solutions = self._best_solutions(keep)
        self.solutions = []
        self.points = set()
        for point in self.random_points(n):
            self.add_point(point)
        return old_solutions

    def add_solutions(self, solutions):
        for s in solutions:
            self.points.add(s[1])
            heappush(self.solutions, s)

    def best(self):
        return self.solutions[0][1]

    def choose_best(self):
        for dim, value in self.solutions[0][1]:
            sim.params.at[self.feature, dim] = value


if __name__ == "__main__":
    from simulation import Simulation
    import mpi

    sim = Simulation.from_files('data/anon_graph_inner_neos_20201110.npz', 'data/sim_features_neos_20201110.csv',
                                seed=2)
    with mpi.futures(sim) as sim:
        if sim is not None:
            climb = Hillclimb(sim, sim.features[0], sources=100, samples=100, seed=1)
            old = climb.random_restart(10)
            # print(f'old: {old}')
            for _ in range(20):
                climb.iterate_stochastic(2, 5)
            for _ in range(10):
                climb.iterate_steep(1, 2)
            # climb.iterate_steep(1, 5)
            # climb.add_solutions(old)
            # climb.iterate_stochastic(1)
            climb.evaluate()
            # print(climb.random_restart())
