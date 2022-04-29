import ast
import os
import pickle
import re
from contextlib import contextmanager
from datetime import datetime
from fnmatch import fnmatch
from os.path import join

import pandas as pd


def parse_file(filename):
    with open(filename, 'r') as file:
        res = {
            'jobid': re.search(r'[_-](\d+)(\.hawk-pbs5-|/)', filename).group(1),
            'filename': os.path.basename(filename),
            'file': filename,
            'date': pd.to_datetime(re.search(r'(\d{8}T\d{4})', filename).group(1)),
        }
        firstline = file.readline()
        if firstline.strip().isdecimal():
            res['nodes'] = int(firstline)
            res['cores'] = int(file.readline())
        else:
            res.update(parse_stats(firstline))

        for line in file:
            res.update(parse_stats(line))
        return res


def parse_stats(s):
    def namespace(s):
        from argparse import Namespace  # noqa: F401

        return vars(eval(s))

    std_fields = (
        ('nodes', int),
        ('mpiprocs', int),
        ('walltime', str),
        ('jobid', lambda x: x.replace(".hawk-pbs5", "")),
        ('date', datetime.fromisoformat),
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
        ('gridsize', int),
        ('rusage', dict, str),
        ('args', ast.literal_eval, namespace),  # second for Namespace
    )

    fields = [
        ('cfeatures', r'(\d+) features, \d+ sources, \d+ samples$', int),
        ('csources', r'\d+ features, (\d+) sources, \d+ samples$', int),
        ('csamples', r'\d+ features, \d+ sources, (\d+) samples$', int),
        ('totaltime', r'Total Time Elapsed: (.+)$', float),
    ]
    fields += [(x, f'{x}: (.+)$', *t) for x, *t in std_fields]

    res = {}
    for f in fields:
        name, regex, *casts = f
        m = re.search(regex, s)
        if m:
            for c in casts:  # Use first successful cast
                try:
                    res[name] = c(m.group(1))
                    break
                except ValueError:
                    pass
            else:
                raise ValueError(f"Could not parse {name}: {m}")
    if 'args' in res:
        res.update(res['args'])
        del res['args']

    renames = {'corrs': 'corr', 'discounts': 'discount'}
    for old, new in renames.items():
        if old in res:
            res[new] = res[old]
            del res[old]

    return res


def parse_files(logfiles):
    r = pd.DataFrame(map(parse_file, logfiles), dtype=object)
    if r.empty:
        print("No files found.")
        return r
    r.set_index('jobid', inplace=True)
    r['nodes'] = r.mpi_size.apply(
        lambda s: s // 48 if s % 48 == 0 else s // 128, convert_dtype=False
    )
    r['total_tweets'] = r.features * r.sources * r.samples
    if 'mean_retweets' in r.columns:
        r['total_retweets'] = r.features * r.sources * r.samples * r.mean_retweets
    #     display(r.isna(column='totaltime'))
    r = r.sort_index().dropna(subset=['topic'])
    aborted = r[r.totaltime.isna()]
    r.drop(aborted.index, inplace=True)
    #     display(aborted)
    print(f'aborted: {aborted.filename}')
    r = r.convert_dtypes()
    return r


def nofilter(_):
    return True


def walk(directory, dirfilter=nofilter, filefilter=nofilter):
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if dirfilter(d)]
        for f in files:
            if filefilter(f):
                yield join(root, f)


def walk_skip_dot(directory, dirfilter=nofilter, filefilter=nofilter):
    yield from walk(
        directory,
        lambda d: d[0] != '.' and dirfilter(d),
        lambda f: not fnmatch(f, '.*') and filefilter(f),
    )


def outfiles(directory):
    yield from walk_skip_dot(directory, filefilter=lambda f: fnmatch(f, '*out*'))


def picklefiles(directory):
    yield from walk_skip_dot(directory, filefilter=lambda f: fnmatch(f, '*.pickle'))


def load_pickle(filename):
    with open(filename, "rb") as file:
        try:
            obj = pickle.load(file)
        except TypeError:
            file.seek(0)
            with old_point():
                obj = pickle.load(file)
    return obj


@contextmanager
def old_point():
    import functools
    from .. import optimization

    @functools.total_ordering
    class Point(dict):  # https://stackoverflow.com/a/1151686
        def __hash__(self):
            return hash(frozenset(self.keys()), frozenset(self.values()))

        def __lt__(self, other):
            return hash(self) < hash(other)

    current = optimization.searchspace.Point
    optimization.searchspace.Point = Point
    yield
    optimization.searchspace.Point = current
