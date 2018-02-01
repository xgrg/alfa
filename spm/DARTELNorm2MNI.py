#!/usr/bin/env python
#coding: utf-8

import sys
import nipype.interfaces.spm as spm
import nipype.pipeline.engine as pe
import os, json
from glob import glob
import os.path as osp
import argparse

def dartel_normalization_to_mni(ff_dir, l2_dir, base_dir, fp, template_fp, subjects, fwhm=0, doit=False):
    ff, rdwi = [], []
    subjects2 = []
    for s in subjects:
        print s
        #try:
        fffp = glob(osp.join(ff_dir, 'u_rc1*%s*_Template.nii'%s))[0]
          # DARTEL flow fields
        print osp.join(l2_dir, 'r%s*_%s.nii'%(s, fp))
        rdwifp = glob(osp.join(l2_dir, 'r%s*_%s.nii'%(s, fp)))[0]
          # Moving images

        rdwi.append(rdwifp)
        ff.append(fffp)
        subjects2.append(s)
        #except:
        #    print s, 'failed'
    print len(rdwi)
    print ff, rdwi

    if not doit: ans = raw_input('Continue ?')

    nm = pe.Node(spm.DARTELNorm2MNI(), name='Norm2MNI')
    nm.inputs.template_file = template_fp
    nm.inputs.flowfield_files = ff
    nm.inputs.fwhm = fwhm
    nm.inputs.apply_to_files = rdwi
    nm.inputs.modulate = False

    w = pe.Workflow(name='DARTELNorm2MNI')
    w.base_dir = base_dir
    w.add_nodes([nm])
    w.run('MultiProc', plugin_args={'n_procs' : 8})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='`DARTELNorm2MNI.py` is used '
        'to apply previously estimated DARTEL flowfields to a set of images.')
    parser.add_argument('source_directory1', help='Directory containing the '
        ' flowfields (u_r*nii)')
    parser.add_argument('source_directory2', help='Directory containing the '
        'realigned images (r*.nii) and the normalized images ((s)wr*.nii)')
    parser.add_argument('target_directory', help='Directory to write the realigned'
        ' sets of images (e.g. r*_c1.nii). (default: source directory)')
    parser.add_argument('--fwhm', default=0, help='Gaussian smoothing kernel to'
        ' use (FWHM) in mm (default:no smoothing)')
    parser.add_argument('subjects', help='Json file containing a list of'\
        'identifiers (allowing to select/discard subjects)')
    parser.add_argument('dartel_template', help='Path to Template_6.nii')
    parser.add_argument('file_pattern', help='Pattern to look for in the filenames '
            '(from the second set)')
    parser.add_argument('--doit', action='store_true', help='Run it without prompting')

    opts = parser.parse_args()
    subjects = json.load(open(opts.subjects))

    dartel_normalization_to_mni(opts.source_directory1, opts.source_directory2,
        opts.target_directory, opts.file_pattern, opts.dartel_template,
        subjects, opts.fwhm, opts.doit)
