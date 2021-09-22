from functools import wraps
from typing import Protocol, Iterator, Tuple, Any, Callable, TypeVar, Optional

import numpy as np
import pandas as pd

from propagation.searchspace import SearchSpace, bisect, middle_value, Point

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


class WithHistory:
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


class WithAllTimeBest:
    def __init__(self, o: FindRoot):
        self.o = o
        self.best = (float("inf"), None)

    def __iter__(self):
        for res in self.o:
            if res[0] <= self.best[0]:
                self.best = res
            yield res

    def best(self):
        return self.best[1]

    def state(self):
        return self.o.state(), self.best


class WithCallback:
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


class FindRootParallel:
    def __init__(self, opts):
        self.opts = opts
        self.active = [iter(o) for o in opts]

    def __iter__(self):
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
        return min(res, key=abs_value)[1]

    def state(self):
        return [o.state() for o in self.opts]


class FindRootMapping:
    def __init__(self, opts):
        self.opts = opts
        self.active = {key: iter(o) for key, o in opts.items()}

    def __iter__(self):
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


def logging(fun):
    """Simple function decorator that prints every function invocation."""

    @wraps(fun)
    def wrapper(*args, **kwargs):
        print(f'calling {fun.__name__}({args},{kwargs})')
        return fun(*args, **kwargs)

    return wrapper


def calculate_retweet_probability(A, sources, p, at_least_one):
    """Return average number of retweeted messages when starting from sources using edge probability p.

    Args:
        A: Adjacency matrix of graph.
        sources: List of source nodes, one per tweet.
        p: Edge probability.

    Returns:
        mean_{x in sources} 1-(1-p)^{deg-(x)}
        This is the expected value of simulate(A, sources, p, depth=1)[1].
    """
    if at_least_one:
        return p
    return sum(1 - (1 - p) ** float(A.indptr[x + 1] - A.indptr[x]) for x in sources) / len(sources)


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


def set_params(point, sim, feature):
    for dim, value in point.items():
        sim.params.at[feature, dim] = value


def optimize_all_features(
        factory,
        sim,
        domain=None,
        statistic='mean_retweets',
        sources=None,
        samples=500,
        explore_current_point=True,
        callback=None,
        history=True,
        num=1,
):
    def optimize(feature):
        return optimize_feature(
            factory,
            sim,
            feature,
            domain=domain,
            statistic=statistic,
            sources=sources,
            samples=samples,
            explore_current_point=explore_current_point,
        )

    if callback is not None:
        optcal = optimize
        optimize = lambda feature: WithCallback(optcal(feature), callback, feature)
    if history:
        opthist = optimize
        optimize = lambda feature: WithHistory(opthist(feature))
    if num is not None:
        optnum = optimize
        optimize = lambda feature: FindRootParallel([optnum(feature) for _ in range(num)])

    return FindRootMapping({feature: optimize(feature) for feature in sim.features})


def optimize_feature(
    factory: FindRootFactory,
    sim,
    feature: str,
    domain: SearchSpace = None,
    statistic='mean_retweets',
    sources=None,
    samples=500,
    explore_current_point=True,
):
    if domain is None:
        domain = {
            'edge_probability': (0.0, 0.3, 0.001),
            'at_least_one': [True, False],
            'discount_factor': (0.0, 1.0, 0.1),
            'corr': (0.0, 1.0, 0.001),
            'max_nodes': range(100, 860, 20),
            'max_depth': [100],
        }
    if not isinstance(domain, SearchSpace):
        domain = SearchSpace(domain)
    objective = single_objective(sim, feature, statistic)
    f = lambda point: sim.simulate(feature, point, sources, samples, True)
    i = domain.to_point(sim.params.astype('object').loc[feature]) if explore_current_point else None
    return factory(f, domain, objective, initial=i, seed=sim.seed.spawn(1)[0])


def combine_results(results):
    mean_retweets, retweet_probability = zip(*results)
    return np.mean(mean_retweets), np.mean(retweet_probability)


def edge_probability_from_retweet_probability(sim, sources=None, eps=1e-5, features=None):
    """Find edge probability for given feature vector (or all if none given)."""
    if features is None:
        features = sim.features

    def fun(feature):
        s = sim._default_sources(sources, feature)
        alo = sim.params.at[feature, 'at_least_one']
        return lambda p: calculate_retweet_probability(
            sim.A,
            s,
            p['retweet_probability'],
            alo,
        )

    def obj(feature):
        goal = sim.stats.at[feature, 'retweet_probability']
        # print('goal:', goal)
        return lambda x: x - goal

    dom = SearchSpace({'retweet_probability': (0, 1, eps)})
    opt = FindRootMapping(
        {feature: MonotoneRoot(fun(feature), dom, obj(feature)) for feature in features}
    )
    for _ in opt:
        pass
    print(opt.best())
    return pd.Series({feature: o['retweet_probability'] for feature, o in opt.best().items()})


def learn(sim, param, goal_stat, lb, ub, eps, sources, samples, features=None):
    def fun(f):
        return lambda x: sim.simulate(
            feature=f, params=x, sources=sources, samples=samples, return_stats=True
        )

    if features is None:
        features = sim.features
    dom = SearchSpace({param: (lb, ub, eps)})
    opt = FindRootMapping(
        {
            feature: [
                WithHistory(
                    MonotoneRoot(
                        fun(feature),
                        dom,
                        single_objective(sim, feature, goal_stat, absolute=False),
                    )
                )
            ]
            for feature in features
        }
    )
    for _ in opt:
        pass
    return pd.Series(opt.best())


def discount_factor_from_mean_retweets(
    self, params=None, sources=None, samples=1000, eps=0.1, features=None
):
    """Find discount factor for given feature vector (or all if none given)."""
    return self.learn(
        param='discount_factor',
        goal_stat='mean_retweets',
        lb=0.0,
        ub=1.0,
        eps=eps,
        params=params,
        sources=sources,
        samples=samples,
        features=features,
    )


def corr_from_mean_retweets(self, params=None, sources=None, samples=1000, eps=0.1, features=None):
    """Find corr for given feature vector (or all if none given)."""
    return self.learn(
        param='corr',
        goal_stat='mean_retweets',
        lb=0.0,
        ub=0.01,
        eps=eps,
        params=params,
        sources=sources,
        samples=samples,
        features=features,
    )
