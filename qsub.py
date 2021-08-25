import subprocess


def qsub(jobname='propagation', nodes=1, procs=128, walltime='01:00:00', mpiargs='', args='sim neos_20201110',
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
        cmd += ['-W', f'depend=afterok:{after}']
    r = subprocess.run(cmd, input=batch, text=True, capture_output=True)
    if r.returncode == 0:
        jobid = r.stdout.strip()
        print(f'submitted job {jobname} as {jobid}')
        return jobid
    print(f'Could not submit {jobname}: {r}')


def dataset(topic, outer=True):
    longtopic = topic + '_20201110'
    outer = 'outer' if outer else 'inner'
    graph = f'data/anon_graph_{outer}_{longtopic}.npz'
    tweets = f'data/sim_features_{longtopic}.csv'
    return f'{topic} --graph {graph} --tweets {tweets}'


def val(topic, params, sources, samples):
    return f'val {dataset(topic)} --sources {sources} --samples {samples} --params out/params-{topic}-{params}.csv'


def optimize(topic, sources, samples):
    return f'optimize {dataset(topic)} --sources {sources} --samples {samples}'


if __name__ == '__main__':
    for topic in ['neos', 'fpoe']:
        args = optimize(topic, sources=64, samples=1000)
        jobid = qsub(args=args, nodes=16, walltime='06:00:00', jobname=f'opt-64-{topic}')
        # args = val(topic, '1334767.hawk-pbs5', sources=256, samples=1000)
        args = val(topic, jobid, sources=256, samples=1000)
        qsub(args=args, nodes=16, walltime='06:00:00', jobname=f'val-256-{topic}', after=jobid)
