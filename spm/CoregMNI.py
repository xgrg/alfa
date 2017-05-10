#!/usr/bin/env python
#coding: utf-8
import sys
import nipype.interfaces.spm as spm
import nipype.pipeline.engine as pe
import os, json
from glob import glob
import os.path as osp
import argparse


def coregister_maps_to_mni(warped_directory, base_dir, mnitpl_fp):
    rdwinh = glob(osp.join(warped_directory, 'wr*10900*_nohdr.nii'))
    print rdwinh, len(rdwinh)
    ans = raw_input('Continue ?')

    nodes = []
    for each in rdwinh:
        s = osp.split(each)[1].split('_')[0][2:]
        print s
        n = pe.Node(spm.Coregister(), name='Coreg%s'%s)
        n.inputs.target = mnitpl_fp
        n.inputs.source = osp.abspath(each)
        nodes.append(n)

    w = pe.Workflow(name='RealignMNI')
    w.base_dir = base_dir
    w.add_nodes(nodes)
    w.run('MultiProc', plugin_args={'n_procs' : 6})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='`CoregMNI.py` coregisters '
        'a set of images (e.g. after the header-cleaning step ((s)wr*.nii)) '
        ' with respect to the MNI template. This results in the generation of a'
        ' final set of realigned images in the MNI space (r(s)wr*.nii).')
    parser.add_argument('warped_directory', help='Directory containing the '
        'normalized header-cleaned images (e.g. (s)wr*.nii)')
    parser.add_argument('target_directory', help='Directory to write the realigned'
        ' sets of images (e.g. r*_c1.nii). (default: source directory)')
    parser.add_argument('--dartel_template', help='Path to Template_6.nii')


    opts = parser.parse_args()
    template = '/home/grg/data/templates/MNI_atlas_templates/MNI_T1.nii' \
        if opts.dartel_template is None else opts.dartel_template

    coregister_maps_to_mni(opts.warped_directory, opts.target_directory, template)
