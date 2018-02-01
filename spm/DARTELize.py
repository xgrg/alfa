#!/usr/bin/env python
#coding: utf-8

import sys
import nipype.interfaces.spm as spm
import nipype.pipeline.engine as pe
import os, json
from glob import glob
import os.path as osp
import argparse
from nipype import logging
log = logging.loggers['workflow']


def create_coregdwi_node(l2, l2b, c1, name):
    n = pe.Node(spm.Coregister(), name=name)
    n.inputs.target = osp.abspath(c1)
    n.inputs.source = osp.abspath(l2)
    n.inputs.apply_to_files = osp.abspath(l2b)
    return n

def create_dartelnorm2mni_node(flow_fields, template_fp, name):
    nm = pe.Node(spm.DARTELNorm2MNI(), name=name)
    nm.inputs.template_file = template_fp
    nm.inputs.flowfield_files = flow_fields
    nm.inputs.fwhm = 0
    ##nm.inputs.apply_to_files = realigned_files
    nm.inputs.modulate = False
    return nm

def create_coregmni_node(name):
    n = pe.Node(spm.Coregister(), name=name)
    mnitpl_fp = '/home/grg/data/templates/MNI_atlas_templates/MNI_T1.nii'
    n.inputs.target = osp.abspath(mnitpl_fp)
    #n.inputs.source = osp.abspath(fp)
    return n

def get_files(c1_dir, l2_dir, l2b_dir, ff_dir, fp, subjects):
    files = {'l2':[], 'l2b': [], 'c1':[], 'ff':[]}
    subjects2 = []
    ff, rdwi = [], []

    for s in subjects:
        try:
            fpatterns = {'l2': osp.join(l2_dir, '%s*%s*.nii'%(s, fp)),
                         'l2b': osp.join(l2b_dir, '%s*%s*.nii'%(s, fp)),
                         #'rc1' : osp.join(rc1_dir, '%s*_rc1.nii'%s),
                         'c1': osp.join(c1_dir, '%s*_c1.nii'%s),
                         #'ff': osp.join(ff_dir, 'u_r*%s*_Template.nii'%s)}
                         'ff': osp.join(ff_dir, 'u_rc1*%s*_Template.nii'%s)}
            for each, filepath in fpatterns.items():
                tfiles = glob(filepath)
                if len(tfiles) != 1:
                    print s, each, 'is ambiguous (%s)'%tfiles
                files[each].append(tfiles[0])
            subjects2.append(s)
        except Exception as e:
            print s, 'failed'

    subjects = subjects2
    print len(subjects), 'subjects ready'
    return files

def create_dartelize_workflows(c1, l2, l2b, ff, template_fp, base_dir, subjects):
    nodes = []
    for i, s in enumerate(subjects):
        n = create_coregdwi_node(l2[i], l2b[i], c1[i], name='CoregDWI_%s'%s)
        nodes.append(n)
    w1 = pe.Workflow(name='CoregDWI')
    w1.base_dir = base_dir
    w1.add_nodes(nodes)
    log.info(nodes)
    log.info('%s nodes'%len(nodes))

    nodes = []
    nm = create_dartelnorm2mni_node(ff, template_fp, name='Norm2MNI')
    nodes.append(nm)
    w2 = pe.Workflow(name='DARTELNorm2MNI')
    w2.base_dir = base_dir
    w2.add_nodes(nodes)

    nodes = []
    for i, s in enumerate(subjects):
        n = create_coregmni_node(name='CoregMNI_%s'%s)
        nodes.append(n)
    log.info(nodes)
    log.info('%s nodes'%len(nodes))

    #w3 = pe.Workflow(name='CoregMNI')
    #w3.base_dir = base_dir
    #w3.add_nodes(nodes)

    return (w1, w2)

def get_outputs(w, name):
    outputs = []
    import gzip, pickle
    import os.path as osp
    for e in w.list_node_names():
        pp = pickle.load(gzip.open(osp.join(w.base_dir, w.name, e,
            'result_%s.pklz'%e), 'rb'))
        output = getattr(pp.outputs, name) #coregistered_files
        if isinstance(output, unicode):
            outputs.append(output)
        elif isinstance(output, list):
            outputs.extend(output)
    return outputs

def copy_to_destdir(maps, destdir):
    import os
    for each in maps:
        cmd = 'cp %s %s'%(each, destdir)
        log.info(cmd)
        os.system(cmd)

def run_workflows(workflows, n_procs=8, copy_outputs=True):
    w1, w2 = workflows
    log.info('Workflow #1')
    w1.run('MultiProc', plugin_args={'n_procs' : n_procs})
    realigned_maps = get_outputs(w1, 'coregistered_files')
    log.info('Realigned maps: %s'%realigned_maps)
    dartel_node = w2.get_node(w2.list_node_names()[0])
    dartel_node.inputs.apply_to_files = realigned_maps
    w2.run('MultiProc', plugin_args={'n_procs' : n_procs})
    #return w2
    normalized_maps = get_outputs(w2, 'normalized_files')

    #if len(w3.list_node_names()) != len(normalized_maps):
    #    log.info('Workflows #2 and #3 have different size of inputs/outputs')

    #for e, m in zip(w3.list_node_names(), normalized_maps):
    #    w3.get_node(e).inputs.source = m
    #w3.run('MultiProc', plugin_args={'n_procs' : n_procs})
    #realigned_normalized_maps = get_outputs(w3, 'coregistered_source')

    if copy_outputs:
        log.warning('Copying outputs to %s'%w1.base_dir)
        #copy_to_destdir(realigned_normalized_maps, w3.base_dir)
        copy_to_destdir(normalized_maps, w1.base_dir)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DARTELize to be documented soon')
    parser.add_argument('c1_directory', help='Directory containing the first'
        ' set of images (e.g. grey maps *_c1.nii)')
    parser.add_argument('rc1_directory', help='Directory containing the set of'
        ' realigned images (e.g. realigned grey maps r*_c1.nii)')
    parser.add_argument('l2_directory', help='Directory containing the second'
            ' set of images (e.g. DWI maps)')
    parser.add_argument('ff_directory', help='flow fields')
    parser.add_argument('template_fp', help='path to Template6')
    parser.add_argument('subjects', help='Json file containing a list of'\
        'subjects (allowing to select/discard participants)')
    parser.add_argument('target_directory', help='workflow dir')
    parser.add_argument('file_pattern', help='Pattern to look for in the filenames '
        '(from the second set)')
    opts = parser.parse_args()

    subjects = json.load(open(opts.subjects))
    files = get_files(opts.c1_directory, opts.rc1_directory, opts.l2_directory,
        opts.ff_directory, opts.file_pattern, subjects)
    log.info(files)
    w1, w2, w3 = create_dartelize_workflows(files['c1'], files['rc1'], files['l2'],
        files['ff'], opts.template_fp, opts.target_directory, subjects)
