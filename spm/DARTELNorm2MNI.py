#!/usr/bin/env python
#coding: utf-8

import sys
import nipype.interfaces.spm as spm
import nipype.pipeline.engine as pe
import os, json
from glob import glob
import os.path as osp
import argparse

def dartel_normalization_to_mni(wd1, wd2, template_fp, subjects, fwhm=0, modulate=False):
    ff, rdwi = [], []
    subjects2 = []
    for s in subjects:
        print s
        try:
            fffp = glob(osp.join(wd1, 'u_r*%s*_c1_Template.nii'%s))[0]
              # DARTEL flow fields
            rdwifp = glob(osp.join(wd2, 'r%s*_mabonlm_nobias.nii'%s))[0]
              # Moving images

            rdwi.append(rdwifp)
            ff.append(fffp)
            subjects2.append(s)
        except:
            print s, 'failed'
    print len(rdwi)
    print ff, rdwi

    ans = raw_input('Continue ?')

    nm = pe.Node(spm.DARTELNorm2MNI(), name='Norm2MNI')
    nm.inputs.template_file = template_fp
    nm.inputs.flowfield_files = ff
    nm.inputs.fwhm = 0
    nm.inputs.apply_to_files = rdwi
    nm.inputs.modulate = False

    w = pe.Workflow(name='DARTELNorm2MNI')
    w.base_dir = od
    w.add_nodes([nm])
    w.run('MultiProc', plugin_args={'n_procs' : 6})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='`DARTELNorm2MNI.py` is used '
        'to apply previously estimated DARTEL flowfields to a set of images.')
    parser.add_argument('source_directory1', help='Directory containing the '
        ' flowfields (u_r*nii)')
    parser.add_argument('source_directory2', help='Directory containing the '
        'realigned images (r*.nii) and the normalized images ((s)wr*.nii)')
    parser.add_argument('--fwhm', default=0, help='Gaussian smoothing kernel to'
        ' use (FWHM) in mm (default:no smoothing)')
    parser.add_argument('subjects', help='Json file containing a list of'\
        'identifiers (allowing to select/discard subjects)')
    parser.add_argument('dartel_template', help='Path to Template_6.nii')

    opts = parser.parse_args()
    subjects = json.load(open(opts.subjects))

    run_dartel(opts.source_directory, opts.target_directory, opts.dartel_template,
        opts.fwhm, subjects)
