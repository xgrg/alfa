import numpy as np
import nibabel as nib
import logging as log
from glob import glob
import pandas as pd
from joblib import Parallel, delayed
import argparse

def __region_values__(map_fp, atlas, func):
    m = np.array(nib.load(map_fp).dataobj)
    n_labels = len(np.unique(atlas))
    label_values = [func(m[atlas==label]) for label in range(1, n_labels)]
    return label_values

def regions_values(atlas_fp, maps_fp, subjects=None, labels=None, func=np.mean,
        n_jobs=7):

    atlas_res = nib.load(atlas_fp).header['pixdim'][:4]
    log.info('Atlas resolution: %s'%str(atlas_res))
    atlas = np.array(nib.load(atlas_fp).dataobj)

    n_labels = len(np.unique(atlas))
    log.info('%s labels in %s'%(n_labels, atlas_fp))


    df = Parallel(n_jobs=n_jobs, verbose=1)(\
        delayed(__region_values__)(maps_fp[i], atlas, func)\
        for i in xrange(len(maps_fp)))

    columns = labels if not labels is None \
            else [str(e) for e in range(1, n_labels)]

    res = pd.DataFrame(df, columns=columns)

    res['subject'] = xrange(len(maps_fp)) if subjects is None else subjects
    res = res.set_index('subject')

    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect values from a label '\
            'map over a set of images and store them in an Excel table.')
    parser.add_argument('files', nargs='+', help='Files to clean the headers of')
    parser.add_argument('--output', required=True,
        help='Excel file to store the output')
    parser.add_argument('--atlas', required=True,
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

    table = regions_values(opts.atlas, opts.files, opts.files, opts.labels,
            getattr(np, opts.function), opts.n_jobs)
    table.to_excel(opts.output)


