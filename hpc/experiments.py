from .sub import sub, configure_host


def dataset(topic, outer=True):
    longtopic = topic + '_20201110'
    outer = 'outer' if outer else 'inner'
    graph = f'data/anon_graph_{outer}_{longtopic}.npz'
    tweets = f'data/sim_features_{longtopic}.csv'
    return f'{topic} --graph {graph} --tweets {tweets}'


def val(topic, sources, samples, params='', corr='', discount=''):
    params = f'--params out/{params}/params-{topic}-{params}.csv' if params else ''
    corr = f'--corr out/{corr}/corr-{topic}-{corr}.csv' if corr else ''
    discount = f'--discount out/{discount}/discount-{topic}-{discount}.csv' if discount else ''
    return (
        f'val {dataset(topic)} --sources {sources} --samples {samples} {params} {corr} {discount}'
    )


def optimize(topic, sources, samples, corr='', discount=''):
    corr = f'--corr out/{corr}/corr-{topic}-{corr}.csv' if corr else ''
    discount = f'--discount out/{discount}/discount-{topic}-{discount}.csv' if discount else ''
    return f'optimize {dataset(topic)} --sources {sources} --samples {samples} {corr} {discount}'


def learn(param, topic, sources, samples, epsilon=0.001):
    return f'learn_{param} {dataset(topic)} --sources {sources} --samples {samples} --epsilon {epsilon}'


def learn_val(topic, repetitions=1):
    djob, discount = sub(
        args=learn('discount', topic, sources=50, samples=100, epsilon=0.01),
        nodes=8,
        walltime='10:00:00',
        jobname=f'discount-{topic}',
    )
    cjob, corr = sub(
        args=learn('corr', topic, sources=50, samples=100, epsilon=0.0001),
        nodes=8,
        walltime='10:00:00',
        jobname=f'corr-{topic}',
    )
    for _ in range(repetitions):
        sub(
            val(topic, sources=64, samples=100, discount=discount, corr=corr),
            nodes=1,
            walltime='00:30:00',
            jobname=f'val-learn-{discount}',
            after=[djob, cjob],
        )
        sub(
            val(topic, sources=256, samples=1000, discount=discount, corr=corr),
            nodes=1,
            walltime='00:30:00',
            jobname=f'val-learn-{discount}',
            after=[djob, cjob],
        )


def learn_opt_val(topic, repetitions=1):
    djob, discount = sub(
        args=learn('discount', topic, sources=256, samples=1000, epsilon=0.01),
        nodes=32,
        walltime='10:00:00',
        jobname=f'discount-{topic}',
    )
    cjob, corr = sub(
        learn('corr', topic, sources=256, samples=1000, epsilon=0.0001),
        nodes=32,
        walltime='10:00:00',
        jobname=f'corr-{topic}',
    )
    jobid, runid = sub(
        optimize(topic, sources=256, samples=1000, corr=corr, discount=discount),
        nodes=128,
        walltime='02:00:00',
        jobname=f'opt-{topic}',
        after=[djob, cjob],
    )
    for _ in range(repetitions):
        sub(
            val(topic, sources=256, samples=1000, params=runid),
            nodes=1,
            walltime='00:30:00',
            jobname=f'val-{runid}',
            after=jobid,
        )


def optimize_val(topic, val_repetitions=1):
    jobid, runid = sub(
        optimize(topic, sources=400, samples=1000),
        nodes=256,
        walltime='24:00:00',
        jobname=f'opt-{topic}',
    )
    for _ in range(val_repetitions):
        sub(
            val(topic, sources=400, samples=1000, params=runid),
            nodes=1,
            walltime='00:30:00',
            jobname=f'val-{runid}',
            after=jobid,
        )


def main():
    for topic in ['neos', 'fpoe', 'socialdistance']:
        # learn_opt_val(topic, repetitions=2)
        # learn_val(topic, 10)
        optimize_val(topic, 3)


if __name__ == '__main__':
    import os

    configure_host(os.environ.get('PROP_HOST', 'local'))
    main()
