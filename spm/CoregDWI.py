#!/usr/bin/env python
#coding: utf-8

import sys
import nipype.interfaces.spm as spm
import nipype.pipeline.engine as pe
import os, json
from glob import glob
import os.path as osp
import argparse

def coregister_maps(c1_dir, rc1_dir, l2_dir, base_dir, subjects, fp='_mabonlm_nobias'):
    '''In directory `wd1` looks for SPM grey maps (*c1.nii), will generate their
    realign version in that same place (r*c1.nii) and will apply that realignment
    to a set of images found in `wd2` containing the pattern `fp` for a given
    list of `subjects`.'''

    files = {'dwi':[],
             't1':[],
             'rt1':[]}
    subjects2 = []

    for s in subjects:
        try:
            fpatterns = {'dwi': osp.join(l2_dir, '%s*%s*.nii'%(s, fp)),
                         't1' : osp.join(rc1_dir, 'r%s*_c1.nii'%s),
                         'rt1': osp.join(c1_dir, '%s*_c1.nii'%s)}
            for each, filepath in fpatterns.items():
                tfiles = glob(filepath)
                if len(tfiles) != 1:
                    print s, each, 'is ambiguous (%s)'%tfiles
                files[each].append(tfiles[0])
            subjects2.append(s)
        except Exception as e:
            print s, 'failed'

    subjects = subjects2
    print len(files['dwi']), 'subjects ready'

    ans = raw_input('Continue ?')

    nodes = []
    for i, s in enumerate(subjects):
        n = pe.Node(spm.Coregister(), name='Coreg%s'%s)
        n.inputs.target = files['rt1'][i]
        n.inputs.source = files['t1'][i]
        n.inputs.apply_to_files = files['dwi'][i]
        nodes.append(n)

    w = pe.Workflow(name='RealignL2onC1')
    w.base_dir = base_dir
    w.add_nodes(nodes)
    w.run('MultiProc', plugin_args={'n_procs' : 6})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='`CoregDWI.py` uses the '\
        'realignment between a set of grey maps (`*c1.nii`) to apply it to a '
        'second set (e.g. DWI or raw T1 maps). Hence it results in two sets of '
        'realigned images.')
    parser.add_argument('c1_directory', help='Directory containing the first'
        ' set of images (e.g. grey maps *_c1.nii)')
    parser.add_argument('rc1_directory', help='Directory containing the set of'
        ' realigned images (e.g. realigned grey maps r*_c1.nii)')
    parser.add_argument('l2_directory', help='Directory containing the second'
            ' set of images (e.g. DWI maps)')
    parser.add_argument('subjects', help='Json file containing a list of'\
        'subjects (allowing to select/discard participants)')
    parser.add_argument('target_directory', help='Directory to write the realigned'
        ' sets of images (e.g. r*_c1.nii). (default: source directory)')
    parser.add_argument('file_pattern', help='Pattern to look for in the filenames '
        '(from the second set)')
    opts = parser.parse_args()

    subjects = json.load(open(opts.subjects))
    coregister_maps(opts.c1_directory, opts.rc1_directory, opts.l2_directory,
        opts.target_directory, subjects, opts.file_pattern)
