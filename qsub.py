import subprocess


def qsub(args='sim neos_20201110', jobname='propagation', nodes=1, procs=128, walltime='01:00:00', mpiargs='',
         after=None, keep_files=True, queue=None, template='mpi.pbs'):
    with open(template, 'r') as f:
        batch = f.read().format(jobname=jobname,
                                nodes=nodes,
                                procs=procs,
                                walltime=walltime,
                                mpiargs=mpiargs,
                                args=args)
    cmd = ['qsub']
    if queue:
        cmd += ['-q', queue]
    if keep_files:
        cmd += ['-koed']
    if after:
        if isinstance(after, list):
            after = ':'.join(after)
        cmd += ['-W', f'depend=afterok:{after}']
    r = subprocess.run(cmd, input=batch, text=True, capture_output=True)
    if r.returncode == 0:
        jobid = r.stdout.strip()
        print(f'submitted job {jobname} as {jobid}')
        runid = f'{jobname}_{jobid}'
        return jobid, runid
    print(f'Could not submit {jobname}: {r}')


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
    return f'val {dataset(topic)} --sources {sources} --samples {samples} {params} {corr} {discount}'


def optimize(topic, sources, samples):
    return f'optimize {dataset(topic)} --sources {sources} --samples {samples}'


def learn(param, topic, sources, samples, epsilon=0.001):
    return f'learn_{param} {dataset(topic)} --sources {sources} --samples {samples} --epsilon {epsilon}'


def learn_val(topic):
    djob, discount = qsub(args=learn('discount', topic, sources=50, samples=100, epsilon=0.01),
                          nodes=16, walltime='10:00:00', jobname=f'discount-{topic}')
    cjob, corr = qsub(learn('corr', topic, sources=50, samples=100, epsilon=0.0001),
                      nodes=16, walltime='10:00:00', jobname=f'corr-{topic}')
    qsub(val(topic, sources=256, samples=1000, discount=discount, corr=corr),
         nodes=16, walltime='00:10:00', jobname=f'val-learn-{topic}', after=[djob, cjob])


def optimize_val(topic):
    # jobid, runid = qsub(optimize(topic, sources=64, samples=1000)+' --corr out/corr-neos-1336746.hawk-pbs5.csv --discount out/discount-neos-1336745.hawk-pbs5.csv',
    jobid, runid = qsub(optimize(topic, sources=64, samples=1000),
                        nodes=32, walltime='24:00:00', jobname=f'opt-{topic}')
    qsub(val(topic, sources=256, samples=1000, params=runid),
         nodes=16, walltime='00:10:00', jobname=f'val-opt-{topic}', after=jobid)


if __name__ == '__main__':
    for topic in ['neos', 'fpoe']:
        # learn_val(topic)
        optimize_val(topic)
