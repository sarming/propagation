import subprocess


def qsub(jobname='propagation', nodes=1, procs=128, walltime='01:00:00', mpiargs='', args='sim neos_20201110',
         keep_files=False, queue=None, template='mpi.pbs'):
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
    subprocess.run(cmd, input=batch, text=True)


def dataset(topic, outer=True):
    topic += '_20201110'
    outer = 'outer' if outer else 'inner'
    graph = f'data/anon_graph_{outer}_{topic}.npz'
    tweets = f'data/sim_features_{topic}.csv'
    return f'--graph {graph} --tweets {tweets}'


if __name__ == '__main__':
    for topic in ['neos']:
        args = f'optimize {topic} --sources=64 --samples=100 {dataset(topic)}'
        qsub(args=args, walltime='02:00:00')
