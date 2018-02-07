import numpy as np
import nibabel as nib
from glob import glob
import pandas as pd
from joblib import Parallel, delayed
import argparse

def roistats_from_map(map_fp, atlas, func):
     m = np.array(nib.load(map_fp).dataobj)
     n_labels = list(np.unique(atlas))
     n_labels.remove(0)
     label_values = [func(m[atlas==label]) for label in n_labels]
     return label_values

def roistats_from_maps(atlas_fp, maps_fp, subjects=None, labels=None,
	func=np.mean, n_jobs=7):

     atlas_im = nib.load(atlas_fp)
     atlas = np.array(atlas_im.dataobj)

     n_labels = list(np.unique(atlas))
     n_labels.remove(0)

     df = Parallel(n_jobs=n_jobs, verbose=1)(\
         delayed(map_values)(maps_fp[i], atlas, func)\
         for i in xrange(len(maps_fp)))

     columns = labels if not labels is None \
             else [int(e) for e in n_labels]
     res = pd.DataFrame(df, columns=columns)

     res['subject'] = xrange(len(maps_fp)) if subjects is None else subjects
     res = res.set_index('subject')

     return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect ROI stats '\
            'over a set of images and store them in an Excel table.')
    parser.add_argument('images', nargs='+', help='Input images')
    parser.add_argument('--output', required=True,
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

    table = roistats_from_maps(opts.roi, opts.images, opts.files, opts.labels,
            getattr(np, opts.function), opts.n_jobs)
    table.to_excel(opts.output)


