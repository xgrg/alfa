#!/usr/bin/env python
#coding: utf-8

from glob import glob
import os.path as osp
import os
import nipype.interfaces.spm as spm
import json
import argparse

def run_dartel(wd, subjects):
    c1, c2 = [], []
    for each in subjects:
        print each
        c1.append(glob(osp.join(wd, 'r%s*_c1.nii'%each))[0])
        c2.append(glob(osp.join(wd, 'r%s*_c2.nii'%each))[0])

    print c1, c2, len(c1), len(c2)
    ans = raw_input('Continue ?')

    print 'DARTEL'
    dartel = spm.DARTEL()
    dartel.inputs.image_files = [c1, c2]
    dartel.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='`DARTEL.py` is used to generate'\
        ' a DARTEL template from previously realigned sets of images (`r*c1.nii`)'\
        ' (NB: running the process in CLI or from Matlab directly may save the '\
        'overhead time due to Python/Nipype).')
    parser.add_argument('directory', help='Directory containing realigned'\
        'grey maps (r*_c1.nii)')
    parser.add_argument('subjects', help='Json file containing a list of'\
        'identifiers (allowing to select/discard subjects)')
    opts = parser.parse_args()
    subjects = json.load(open(opts.subjects))
    run_dartel(opts.directory, subjects)
