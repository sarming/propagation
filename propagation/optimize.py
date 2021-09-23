import time

import numpy as np
import pandas as pd
from optimization import (
    Bayesian,
    DictFindRoot,
    FindRootFactory,
    FindRootParallel,
    GridSearch,
    MonotoneRoot,
    SearchSpace,
    SingleHillclimb,
    WithAllTimeBest,
    WithCallback,
    WithHistory,
    WithTimeout,
)


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


def set_params(best_points, sim):
    for feature, point in best_points.items():
        for dim, value in point.items():
            sim.params.at[feature, dim] = value


def optimize_feature(
    factory: FindRootFactory,
    sim,
    feature: str,
    domain: SearchSpace = None,
    statistic='mean_retweets',
    sources=None,
    samples=500,
    explore_current_point=True,
    absolute=True,
    history=False,
    callback=None,
    num=None,
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
    objective = single_objective(sim, feature, statistic, absolute)
    f = lambda point: sim.simulate(feature, point, sources, samples, True)
    i = domain.to_point(sim.params.astype('object').loc[feature]) if explore_current_point else None
    if callback is not None:
        factory = WithCallback.wrap(factory, callback, feature)
    if history:
        factory = WithHistory.wrap(factory)
    if num is not None:
        seeds = sim.seed.spawn(num)
        return WithAllTimeBest(
            FindRootParallel(factory(f, domain, objective, initial=i, seed=seed) for seed in seeds)
        )
    return factory(f, domain, objective, initial=i, seed=sim.seed.spawn(1)[0])


def optimize_all_features(factory, sim, *args, features=None, **kwargs):
    if features is None:
        features = sim.features

    return DictFindRoot(
        {feature: optimize_feature(factory, sim, feature, *args, **kwargs) for feature in features}
    )


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
        return lambda p: calculate_retweet_probability(sim.A, s, p['retweet_probability'], alo)

    def obj(feature):
        goal = sim.stats.at[feature, 'retweet_probability']
        # print('goal:', goal)
        return lambda x: x - goal

    dom = SearchSpace({'retweet_probability': (0, 1, eps)})
    opt = DictFindRoot(
        {feature: MonotoneRoot(fun(feature), dom, obj(feature)) for feature in features}
    )
    for _ in opt:
        pass
    return pd.Series({feature: o['retweet_probability'] for feature, o in opt.best().items()})


def learn(sim, param, goal_stat, lb, ub, eps, sources, samples, features=None):
    def fun(f):
        return lambda x: sim.simulate(
            feature=f, params=x, sources=sources, samples=samples, return_stats=True
        )

    if features is None:
        features = sim.features
    dom = SearchSpace({param: (lb, ub, eps)})
    opt = DictFindRoot(
        {
            feature: WithAllTimeBest(
                MonotoneRoot(
                    fun(feature),
                    dom,
                    single_objective(sim, feature, goal_stat, absolute=False),
                )
            )
            for feature in features
        }
    )
    for _ in opt:
        pass
    return pd.Series({feature: o[param] for feature, o in opt.best().items()})


def discount_from_mean_retweets(sim, sources=None, samples=1000, eps=0.1, features=None):
    """Find discount factor for given feature vector (or all if none given)."""
    return learn(
        sim,
        param='discount_factor',
        goal_stat='mean_retweets',
        lb=0.0,
        ub=1.0,
        eps=eps,
        sources=sources,
        samples=samples,
        features=features,
    )


def corr_from_mean_retweets(sim, sources=None, samples=1000, eps=0.1, features=None):
    """Find corr for given feature vector (or all if none given)."""
    return learn(
        sim,
        param='corr',
        goal_stat='mean_retweets',
        lb=0.0,
        ub=0.01,
        eps=eps,
        sources=sources,
        samples=samples,
        features=features,
    )


def gridsearch(sim, sources=None, samples=1000):
    dom = {
        # 'edge_probability': (0., 0.3, .001),
        'discount_factor': (0.0, 1.0, 0.01),
        'corr': (0.0, 0.005, 0.0001),
    }
    print(f'grid: {dom}')

    opts = optimize_all_features(
        GridSearch, sim, domain=dom, sources=sources, samples=samples, explore_current_point=False
    )
    for _ in opts:
        # print(_)
        pass

    return opts.best(), opts.state()


def bayesian(sim, sources=None, samples=1000):
    dom = {
        # 'edge_probability': (0., 0.3, .001),
        'discount_factor': (0.0, 1.0, 0.01),
        'corr': (0.0, 0.005, 0.0001),
    }
    print(f'bayes: {dom}')

    opts = optimize_all_features(
        WithTimeout.wrap(Bayesian, 20),
        sim,
        domain=dom,
        sources=sources,
        samples=samples,
        explore_current_point=True,
        history=True,
        # callback=lambda res, o, feature: print(f"{feature}: {res}"),
        num=1,
    )
    for i, res in zip(range(10000), opts):
        print(i, res)
    return opts.best(), opts.state()


def hillclimb(sim, num=None, sources=None, samples=1000):
    dom = {
        # 'edge_probability': (0., 0.3, .001),
        'discount_factor': (0.0, 1.0, 0.01),
        'corr': (0.0, 0.005, 0.0001),
    }
    print(f'hill: {dom}')

    opts = optimize_all_features(
        SingleHillclimb,
        sim,
        domain=dom,
        sources=sources,
        samples=samples,
        explore_current_point=True,
        num=num,
    )
    for i, _ in zip(range(10), opts):
        # print(_)
        pass
    return opts.best(), opts.state()

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

    return {feature: o.state() for feature, o in opts.items()}


# sourcery skip: hoist-if-from-if, merge-nested-ifs, remove-redundant-if
if __name__ == "__main__":
    from propagation.simulation import Simulation

    sim = Simulation.from_files(
        'data/anon_graph_inner_neos_20201110.npz', 'data/sim_features_neos_20201110.csv', seed=3
    )
    # with mpi.futures(sim) as sim:
    if True:
        if sim is not None:
            bayesian(sim, sources=1, samples=1)
            # hillclimb(sim, sources=2, samples=10, num=2)
