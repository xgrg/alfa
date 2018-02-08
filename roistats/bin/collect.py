#! /usr/bin/env python
import argparse
import sys
import string
import os.path as op
import numpy as np
sys.path.append(op.join(op.expanduser('~'), 'git', 'alfa'))
from roistats import collect

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect ROI stats '\
            'over a set of images and store them in an Excel table.')
    parser.add_argument('images', nargs='+', help='Input images')
    parser.add_argument('-o', required=True,
        help='Excel file to store the output')
    parser.add_argument('--roi', required=True,
        help='Atlas (or any label volume) containing the reference regions')
    parser.add_argument('--labels',
        help='Textfile with label lookup table')
    parser.add_argument('--n_jobs', default=-1,
        help='Number of parallel jobs')
    parser.add_argument('--function', default='mean',
        help = 'numpy function used to get values')
    parser.add_argument('--verbose', action='store_true',
        help='Be verbose')
    opts = parser.parse_args()

    if opts.verbose:
        log.basicConfig(level=log.INFO)

    n_jobs = string.atoi(opts.n_jobs)
    table = collect.roistats_from_maps(opts.images, opts.roi, opts.images, opts.labels,
            getattr(np, opts.function), n_jobs)
    table.to_excel(opts.o)


