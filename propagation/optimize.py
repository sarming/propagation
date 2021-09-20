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


class WithHistory:
    def __init__(self, o: FindRoot):
        self.o = o
        self.history = []

    def __iter__(self):
        for res in self.o:
            self.history.append(res)
            yield res

    def best(self):
        return min((abs(o), p) for o, p in self.history)[1]

    def state(self):
        return self.o.state(), self.history


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
        self.opts = [iter(o) for o in opts]

    def __iter__(self):
        return self

    def __next__(self):
        return min(o.__next__() for o in self.opts)

    def state(self):
        return [o.state() for o in self.opts]


class FindRootCollection:
    def __init__(self, opts):
        self.active = {key: [iter(o) for o in os] for key, os in opts.items()}
        self.opts = opts

    def __iter__(self):
        return self

    def __next__(self):
        next_active = {}
        res = {}
        for key, opts in self.active.items():
            next_opts = []
            res_opts = []
            print(key)
            for o in opts:
                try:
                    r = o.__next__()
                    next_opts.append(o)
                    res_opts.append(r)
                except StopIteration:
                    pass
            next_active[key] = next_opts
            res[key] = res_opts
        self.active = next_active

        if not any(self.active.values()):
            raise StopIteration()

        return res

        # return {key: [o.__next__() for o in opts] for key, opts in self.opts.items()}

    def best(self):
        return {key: [o.best() for o in opts] for key, opts in self.opts.items()}

    def state(self):
        return {key: [o.state() for o in opts] for key, opts in self.opts.items()}


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
    callback,
    sim,
    domain=None,
    statistic='mean_retweets',
    sources=None,
    samples=500,
    explore_current_point=True,
    num=1,
):
    def optimize(feature):
        return WithHistory(
            WithCallback(
                optimize_feature(
                    factory,
                    sim,
                    feature,
                    domain=domain,
                    statistic=statistic,
                    sources=sources,
                    samples=samples,
                    explore_current_point=explore_current_point,
                ),
                callback,
                feature,
            )
        )

    return FindRootCollection(
        {feature: [optimize(feature) for _ in range(num)] for feature in sim.features}
    )


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
        print('goal:', goal)
        return lambda x: x - goal

    dom = SearchSpace({'retweet_probability': (0, 1, eps)})
    opt = FindRootCollection(
        {feature: [MonotoneRoot(fun(feature), dom, obj(feature))] for feature in features}
    )
    for _ in opt:
        pass
    print(opt.best())
    return pd.Series({feature: o[0]['retweet_probability'] for feature, o in opt.best().items()})


def learn(sim, param, goal_stat, lb, ub, eps, sources, samples, features=None):
    if features is None:
        features = sim.features
    dom = SearchSpace({param: (lb, ub, eps)})
    opt = FindRootCollection(
        {
            feature: [
                WithHistory(
                    MonotoneRoot(
                        lambda point: sim.simulate(feature, point, sources, samples, True),
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


# @timecall
def learn(self, param, goal_stat, lb, ub, eps, params, sources, samples, features):
    def set_param(value, feature):
        p = self._default_params(params, feature)
        p[param] = value
        return p

    def fun(feature):
        return lambda x: list(
            self.simulator(
                A=self.A,
                params=set_param(x, feature),
                sources=self._default_sources(sources, feature),
                samples=samples,
                seed=self.seed.spawn(1)[0],
            )
        )[0 if goal_stat == 'mean_retweets' else 1]

    if features is None:
        features = self.features
    return pd.Series(
        (
            invert_monotone(
                fun=fun(f),
                goal=self.stats.at[f, goal_stat],
                lb=lb,
                ub=ub,
                eps=eps,
                logging=True,
            )
            for f in features
        ),
        index=features,
    )


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
