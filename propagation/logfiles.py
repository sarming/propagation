import ast
import os
import re
from fnmatch import fnmatch
from os.path import join

import pandas as pd


def parse_file(filename):
    with open(filename, 'r') as file:
        res = {
            'jobid': re.search('[_-](\d+\.hawk-pbs5)', filename).group(1),
            'filename': os.path.basename(filename),
            'file': filename,
        }
        firstline = file.readline()
        if firstline.strip().isdecimal():
            res['nodes'] = int(firstline)
            res['cores'] = int(file.readline())
        else:
            res.update(parse_stats(firstline))

        for l in file:
            res.update(parse_stats(l))
        return res


def parse_stats(s):
    std_fields = (
        ('nodes', int),
        ('mpiprocs', int),
        ('walltime', str),
        ('jobid', str),
        ('date', str),
        ('topic', str),
        ('code_version', str),
        ('argv', str),
        ('mpi_size', int),
        ('mae_mean_retweets', float),
        ('mape_mean_retweets', float),
        ('mae_retweet_probability', float),
        ('mape_retweet_probability', float),
        ('setuptime', float),
        ('runtime', float),
        ('totaltime', float),
        ('seed', int),
        ('grid', ast.literal_eval),
        ('opt', ast.literal_eval),
        ('mae', float),
        ('mape', float),
        ('wmape', float),
    )

    fields = [
        ('cfeatures', '(\d+) features, \d+ sources, \d+ samples$', int),
        ('csources', '\d+ features, (\d+) sources, \d+ samples$', int),
        ('csamples', '\d+ features, \d+ sources, (\d+) samples$', int),
        ('totaltime', 'Total Time Elapsed: (.+)$', float),
    ]
    fields += [(x, f'{x}: (.+)$', t) for x, t in std_fields]

    res = {}
    for f in fields:
        m = re.search(f[1], s)
        if m:
            res[f[0]] = f[2](m.group(1))

    m = re.search('args: (.+)?', s)
    if m:
        try:
            args = ast.literal_eval(m.group(1))
        except ValueError:
            from argparse import Namespace
            args = vars(eval(m.group(1)))
            pass
        finally:
            res.update(args)

    return res


def parse_files(logfiles):
    r = pd.DataFrame(map(parse_file, logfiles), dtype=object)
    r.set_index('jobid', inplace=True)
    r['nodes'] = r.mpi_size // 128
    #    r['total_tweets'] = r.featurabses * r.sources * r.samples
    #    r['total_retweets'] = r.features * r.sources * r.samples * r.mean_retweets
    #     display(r.isna(column='totaltime'))
    r = r.sort_index().dropna(subset=['topic'])
    aborted = r[r.totaltime.isna()]
    r.drop(aborted.index, inplace=True)
    #     display(aborted)
    print(f'aborted: {aborted.filename}')
    r = r.infer_objects()
    return r


def outfiles(directory):
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d[0] != '.']  # remove dotdirs
        for f in files:
            if fnmatch(f, '*out*') and not fnmatch(f, '.*'):
                yield join(root, f)
