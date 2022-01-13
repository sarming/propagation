#!/usr/bin/env python
import argparse
import functools
import os
import subprocess


def qsub_cmd(after, queue):
    cmd = ['qsub']
    if queue:
        cmd += ['-q', queue]
    if after:
        if isinstance(after, list):
            after = ':'.join(after)
        cmd += ['-W', f'depend=afterok:{after}']
    return cmd, lambda x: x


def sbatch_cmd(after, queue):
    cmd = ['sbatch']
    if queue:
        cmd += ['-p', queue]
    if after:
        if isinstance(after, list):
            after = ':'.join(after)
        cmd += ['-d', f'afterok:{after}']
    return cmd, lambda x: x.replace("Submitted batch job ", "")


def bash(after, queue):
    return ['bash'], lambda x: x


def sub(
    args='sim neos_20201110',
    exe='bin/run',
    jobname='propagation',
    mpiargs='',
    nodes=1,
    procs=1,
    walltime='01:00:00',
    template='local.sh',
    queue=None,
    after=None,
    sub_cmd=bash,
):
    template = os.path.join(os.path.dirname(__file__), template)

    with open(template, 'r') as f:
        batch = f.read().format(
            args=args,
            exe=exe,
            jobname=jobname,
            mpiargs=mpiargs,
            nodes=nodes,
            procs=procs,
            walltime=walltime,
        )
    cmd, get_jobid = sub_cmd(after, queue)
    r = subprocess.run(cmd, input=batch, text=True, capture_output=True)
    if r.returncode == 0:
        jobid = get_jobid(r.stdout.strip())
        print(f'submitted job {jobname} as {jobid}')
        runid = f'{jobname}_{jobid}'
        # print(r)
        return jobid, runid
    raise RuntimeError(f'Could not submit {jobname!r}: {r}')


default_config = dict(
    args='sim neos_20201110',
    jobname='propagation',
    nodes=1,
    walltime='00:10:00',
    mpiargs='',
    procs=1,
    after=None,
    queue=None,
    template='local.sh',
    sub_cmd=bash,
    exe='bin/run',
)


host_configs = {
    'local': {'template': 'local.sh', 'sub_cmd': bash, 'procs': None},  # Use nodes as procs
    'hawk': {'template': 'hawk.pbs', 'sub_cmd': qsub_cmd, 'procs': 128},
    'mach2': {
        'template': 'mach2.pbs',
        'sub_cmd': qsub_cmd,
        'queue': 'f2800',
        'procs': None,  # Use nodes as procs
    },
    'supermuc': {
        'template': 'supermuc.sh',
        'sub_cmd': sbatch_cmd,
        'queue': 'test',
        'procs': 48,
    },
    'training': {
        'template': 'training.sh',
        'sub_cmd': sbatch_cmd,
        'queue': 'hidalgo',
        'procs': 32,
    },
}
host_configs = {k: {**default_config, **v} for k, v in host_configs.items()}
current_config = default_config


def default_wrapper(defaults, wrapped):
    def wrapper(*args, **kwargs):
        args = dict(zip(defaults.keys(), args))
        return wrapped(**{**defaults, **args, **kwargs})

    return functools.update_wrapper(wrapper, wrapped)


def configure_host(host):
    global current_config, sub
    current_config = host_configs[host]
    if 'old_sub' not in vars(configure_host):
        configure_host.old_sub = sub
    sub = default_wrapper(current_config, configure_host.old_sub)


def parse_args(config):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--host',
        choices=host_configs.keys(),
        default=os.environ.get('PROP_HOST', os.environ.get('PROP_HOST', 'local')),
    )
    parser.add_argument('--jobname', help="jobname", default=config['jobname'])
    parser.add_argument('-t', '--walltime', default=config['walltime'])
    parser.add_argument('-n', '--nodes', type=int, default=config['nodes'])
    parser.add_argument('-q', '--queue', default=config['queue'])
    parser.add_argument(
        '--profile',
        help="profile run",
        action="store_const",
        const="bin/profile",
        default="bin/run",
        dest="exe",
    )
    return parser.parse_known_args()


def main():
    args, _ = parse_args(default_config)
    configure_host(args.host)
    args, run_args = parse_args(current_config)
    run_args = run_args or [current_config['args']]
    _, j = sub(
        args=' '.join(run_args),
        jobname=args.jobname,
        nodes=args.nodes,
        walltime=args.walltime,
        queue=args.queue,
        exe=args.exe,
    )
    print(f'outdir: out/{j}')


if __name__ == '__main__':
    main()
