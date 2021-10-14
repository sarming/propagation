import pickle

import pandas as pd

from optimization.searchspace import SearchSpace
from . import optimize, tree
from .simulation import Simulation


def run(sim: Simulation, args):
    cmds = {
        'learn_corr': learn_corr,
        'learn_discount': learn_discount,
        'optimize': opt,
        'sim': simstats,
        'simtweets': simtweets,
        'trees': simtweets,
        'val': val,
    }
    cmds[args.command](sim, args)


def learn_corr(sim: Simulation, args):
    corr = optimize.corr_from_mean_retweets(sim, samples=args.samples, eps=args.epsilon)
    corr.to_csv(f'{args.outdir}/corr-{args.topic}-{args.runid}.csv')
    sim.params['corr'] = corr
    sim.params.to_csv(f'{args.outdir}/params-{args.topic}-{args.runid}.csv')


def learn_discount(sim: Simulation, args):
    discount = optimize.discount_from_mean_retweets(sim, samples=args.samples, eps=args.epsilon)
    discount.to_csv(f'{args.outdir}/discount-{args.topic}-{args.runid}.csv')
    sim.params['discount_factor'] = discount
    sim.params.to_csv(f'{args.outdir}/params-{args.topic}-{args.runid}.csv')


def opt(sim: Simulation, args):
    dom = {
        'discount_factor': (0.0, 1.0, 200 * args.epsilon),  # = 0.2 * (eps / 0.001)
        'corr': (0.0, 0.005, args.epsilon),  # = 0.001 * (eps / 0.001)
    }
    dom = SearchSpace(dom)

    print("grid:", dom.bounds)
    print("gridsize:", dom.size(), flush=True)

    best, state = optimize.gridsearch(
        sim,
        sources=None if args.sources < 1 else args.sources,
        samples=args.samples,
        dom=dom,
    )
    optimize.set_params(best, sim)
    sim.params.to_csv(f'{args.outdir}/params-{args.topic}-{args.runid}.csv')
    with open(f'{args.outdir}/optimize-{args.topic}-{args.runid}.pickle', 'bw') as f:
        pickle.dump((best, state), f)
    # last history element in first optimization
    # objective = pd.Series({k: o[0][1][-1] for k, o in opts.items()})
    # real = sim.stats.mean_retweets
    # sim = real + objective
    # print(f'mae: {mae(sim, real)}')
    # print(f'mape: {mape(sim, real)}')
    # print(f'wmape: {wmape(sim, real)}')


def simstats(sim: Simulation, args):
    r = agg_statistics(
        (feature, sim.simulate(feature, sources=args.sources, samples=args.samples))
        for feature in sim.sample_feature(args.features)
    )
    r.to_csv(f'{args.outdir}/results-sim-{args.topic}-{args.runid}.csv')
    print(r)


def simtweets(sim: Simulation, args):
    explode = explode_trees if args.command == "trees" else explode_tweets
    r = sim.run(num_features=args.features, num_sources=args.sources, samples=args.samples)
    r = explode(r)
    r.to_csv(f'{args.outdir}/results-{args.command}-{args.topic}-{args.runid}.csv', index=False)
    print(r)


def val(sim: Simulation, args):
    r = agg_statistics(
        (feature, sim.simulate(feature, sources=args.sources, samples=args.samples))
        for feature in sim.features
    )
    # for feature in [('0000', '0000')])

    # assert r.index.equals(sim.stats.index)
    # r = r.reindex(index=sim.features)
    r.columns = pd.MultiIndex.from_product([['sim'], r.columns])
    r[('real', 'mean_retweets')] = sim.stats.mean_retweets
    r[('real', 'retweet_probability')] = sim.stats.retweet_probability

    r.to_csv(f'{args.outdir}/results-val-{args.topic}-{args.runid}.csv')
    # r = pd.read_csv(..., header=[0,1], index_col=[0,1])

    pretty = r.swaplevel(axis=1).sort_index(axis=1).drop(columns='tweets')
    pretty.index = r.index.to_flat_index()
    print(pretty)

    def print_error(measure, stat, real=r.real, sim=r.sim):
        print(f"{measure.__name__}_{stat}: {measure(sim[stat], real[stat])}")

    for stat in ['retweet_probability', 'mean_retweets']:
        for measure in [mae, mape, wmape]:
            print_error(measure, stat)


# Helper functions:


def agg_statistics(feature_results):
    """Aggregate statistics by feature.

    Args:
        feature_results: iterable of (feature, (mean_retweets, retweet_probability))

    Returns:
        DataFrame with aggregated number of tweets, mean_retweets and retweet_probability.

    """
    r = pd.DataFrame(feature_results, columns=['feature', 'results'])
    r[['author_feature', 'tweet_feature']] = pd.DataFrame(r['feature'].tolist())
    r[['mean_retweets', 'retweet_probability']] = pd.DataFrame(r['results'].tolist())
    return r.groupby(['author_feature', 'tweet_feature']).agg(
        tweets=('feature', 'size'),  # TODO: multiply with authors * samples
        mean_retweets=('mean_retweets', 'mean'),
        retweet_probability=('retweet_probability', 'mean'),
    )


def explode_tweets(tweet_results) -> pd.DataFrame:
    # input is of the form [(feature1, results1), (feature2, results2), ...]
    # where each results is of the form [(author1, [retweet11, retweeet12, ...]), (author2, [retweets21, ...]), ...]
    # (list are generators)

    r = pd.DataFrame(tweet_results, columns=['feature', 'results'])
    r[['author_feature', 'tweet_feature']] = pd.DataFrame(r['feature'].tolist())

    r['results'] = r['results'].apply(list)
    r = r.explode('results', ignore_index=True)

    r[['author', 'retweets']] = pd.DataFrame(r['results'].tolist())

    r['retweets'] = r['retweets'].apply(list)
    r = r.explode('retweets', ignore_index=True)

    return r[['author', 'author_feature', 'tweet_feature', 'retweets']]


def explode_trees(results) -> pd.DataFrame:
    with tree.propagation_tree():
        r = explode_tweets(results)
    r.rename(columns={'retweets': 'tree'}, inplace=True)
    hist = r['tree'].apply(lambda x: tree.depth_histogram(tree.from_dict(x)))
    r = r.join(hist.fillna(0))
    return r


def mae(sim, real=0.0):
    return (sim - real).abs().mean()


def mape(sim, real):
    return ((sim - real) / real).abs().mean()


def wmape(sim, real):
    return ((sim - real) / real.mean()).abs().mean()
