#! /usr/bin/env python
import os, json
from glob import glob
import os.path as osp
import numpy as np
import nibabel as nib
from nilearn import image
import argparse
import logging as log


if __name__ == '__main__':
    log.basicConfig(level=log.INFO)
    parser = argparse.ArgumentParser(description='`remove_csf.py` looks for '\
        'voxels with a CSF-probability higher than a given threshold and gives '\
        'them 0 in a given image (removes any voxel from skull and scalp). '\
        'Filenames of segmentation maps will be guessed '\
        'from the grey matter mask file.'\
        'Please make sure both are coregistered.')
    parser.add_argument('image', help='image to remove voxels from')
    parser.add_argument('grey_matter_mask', help='grey matter mask')
    parser.add_argument('destdir', help='destination folder')
    parser.add_argument('--threshold', type=float, help='threshold of CSF-probability',
            default=0.0)
    opts = parser.parse_args()

    mdfp = opts.image #glob('/home/grg/dartel_final/native/%s*_t1space.nii'%s)[0]
    md = np.asarray(nib.load(mdfp).dataobj)
    log.info('Image has dimensions: %s'%str(md.shape))
    for i, n in zip([3,4,5], ['CSF', 'Skull', 'Scalp']):
       csffp = opts.grey_matter_mask.replace('c1', 'c%s'%i) #glob('/home/grg/data/AmylStaging/%s/T1/%s*spm_c%s.nii'%(s,s,i))[0]
       if not osp.isfile(csffp):
           raise IOError('%s not found'%csffp)
       csf = np.asarray(nib.load(csffp).dataobj)
       log.info('%s map filename was found at: %s (dimensions: %s)'
               %(n, csffp, str(csf.shape)))
       threshold = opts.threshold if i==3 else 0.0
       md[csf > threshold] = 0
    fp = osp.join(opts.destdir,
            '%s_wo_csf.nii'%osp.basename(mdfp).split('.nii')[0])
    image.new_img_like(mdfp, md).to_filename(fp)
    log.info('Succesfully written %s'%fp)
