#! /usr/bin/env python

# SPM12 Nipype pipeline
# =====================


# Atlases from SPM (first volume only ~ grey matter map)
# ======================================================
# Since you can't select a volume from a multivolume nifti as a
# nipype.interfaces.spm.Coregister input (as opposed to with SPM), this ICBM
# atlas version contains the grey matter map only (after splitting the original
# one).
# ex: fslsplit /usr/local/MATLAB/R2014a/toolbox/spm12/toolbox/DARTEL/icbm152.nii
#     mv vol0000.nii.gz icbm152_c1.nii.gz
#     gunzip icbm152_c1.nii.gz

icbm_atlas_fp = '/home/grg/data/AmylStaging/icbm152_c1.nii'
mni_atlas_fp = '/usr/local/MATLAB/R2014a/toolbox/spm12/canonical/avg152T1.nii'

def _get_diskitem_(source):
    import neuroHierarchy, neuroProcesses
    dsk = neuroHierarchy.databases.createDiskItemFromFileName(source)
    return dsk

def _get_diskitems_(source):

    import neuroHierarchy, neuroProcesses
    types = ["ALFA Denoised Nobias SPM Grey matter",
             "ALFA Denoised Nobias SPM White matter",
             "ALFA Denoised Nobias SPM CSF",
             "ALFA Denoised Nobias SPM Skull",
             "ALFA Denoised Nobias SPM Scalp",
             "ALFA Denoised Nobias SPM DARTEL-imported Grey matter",
             "ALFA Denoised Nobias SPM DARTEL-imported White matter"
             ]
    s = []
    dsk = _get_diskitem_(source)
    for t in types:
        w = neuroHierarchy.WriteDiskItem(t, neuroProcesses.getAllFormats())
        fp = w.findValue(dsk)
        s.append(fp)

    log.info('Resolved files: %s'%str(zip(types, s)))
    return s

def _get_subject_(source):
    dsk = _get_diskitem_(source)
    return dsk.get('subject')

def create_coregister_node(source, target, name):
    from nipype.pipeline.engine import Node
    import nipype.interfaces.spm as spm
    coreg = spm.Coregister()
    coreg.inputs.jobtype = 'estimate'
    coreg.inputs.target = target
    coreg.inputs.source = source
    return Node(coreg, name)

def create_spm12_node(source, name):
    from nipype.pipeline.engine import Node
    import nipype.interfaces.spm as spm
    seg = spm.NewSegment()
    seg.inputs.channel_files = source
    tpm = '/usr/local/MATLAB/R2014a/toolbox/spm12/toolbox/Seg/TPM.nii'
    tissue1 = ((tpm, 1), 2, (True,True), (False, False))
    tissue2 = ((tpm, 2), 2, (True,True), (False, False))
    tissue3 = ((tpm, 3), 2, (True,False), (False, False))
    tissue4 = ((tpm, 4), 2, (True,False), (False, False))
    tissue5 = ((tpm, 5), 2, (True,False), (False, False))
    seg.inputs.tissues = [tissue1, tissue2, tissue3, tissue4, tissue5]
    return Node(seg, name)

def create_nodes(source, subject):
    nodes = []
    coreg1 = create_coregister_node(source, target=icbm_atlas_fp, name='coreg_icbm_%s'%subject)
    nodes.append(coreg1)
    coreg2 = create_coregister_node(source, target=mni_atlas_fp, name='coreg_mni_%s'%subject)
    nodes.append(coreg2)

    seg = create_spm12_node(source, 'spm12_%s'%subject)
    nodes.append(seg)
    return nodes


def create_workflow(sources, subjects, basedir=None):
    import os.path as osp
    import tempfile
    from nipype.pipeline.engine import Workflow, Node
    if len(sources) != len(subjects):
        raise Exception('Input files and subjects should be of equal size.')
    wf_name = 'spm12_%s'%subjects[0] if len(subjects) == 1 else 'spm12'
    if len(sources) == 1 and basedir is None:
        wf_basedir = osp.dirname(sources[0])
    elif not basedir is None:
        wf_basedir = basedir
    else:
        wf_basedir = tempfile.mkdtemp()

    w = Workflow(wf_name, base_dir = wf_basedir)
    nodes = []
    for subject, source in zip(subjects, sources):
        nodes.extend(create_nodes(source, subject))
    w.add_nodes(nodes)

    for i in range(0, len(nodes), 3):
        w.connect(nodes[i], 'coregistered_source', nodes[i+1], 'source')
        w.connect(nodes[i+1], 'coregistered_source', nodes[i+2], 'channel_files')
    return w


if __name__ == '__main__':
    import argparse
    import logging as log
    import os
    from os.path import abspath as ap

    log.basicConfig(level=log.INFO)
    parser = argparse.ArgumentParser(prog='nipype_spm12',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='''Run SPM12 segmentation pipeline.
            If INPUT is an Axon database:
              - will guess the subject identifier
              - will copy generated outputs to the Axon database
            If INPUT is a solitary file:
              - SUBJECT argument is mandatory''')

    parser.add_argument('input', type=str,
            help='NIfTI image to run SPM12 segmentation pipeline on')
    parser.add_argument('--subject', type=str,
            help='subject identifier to name Nipype nodes after')
    parser.add_argument('--destdir', type=str,
            help='destination folder where to write/run the jobs.')
    args = parser.parse_args()
    source = args.input
    basedir = args.destdir
    log.info('Workflow base directory set to: %s'%basedir)

    if not args.subject:
        from brainvisa import axon
        axon.initializeProcesses()
        subject = _get_subject_(ap(source))
    else:
        subject = args.subject

    log.info('Processing file %s (subject: %s)'%(ap(source), subject))
    w = create_workflow([ap(source)], [subject], basedir=basedir)
    res = w.run()

    if not args.subject:
        spm_node_name = [e for e in w.list_node_names() if 'spm12' in e][0]
        log.info('SPM node name: %s'%spm_node_name)
        spm_node = w.get_node(spm_node_name)
        import gzip, pickle
        import os.path as osp
        pp = pickle.load(gzip.open(osp.join(w.base_dir, w.name, spm_node_name,
                'result_%s.pklz'%spm_node_name), 'rb'))

        outputs = [e for e in pp.outputs.native_class_images]
        outputs.extend(pp.outputs.dartel_input_images)
        s = _get_diskitems_(ap(source))
        for fp1, fp2 in zip(outputs, s):
            cmd = 'cp %s %s'%(fp1[0], fp2)
            log.info(cmd)
            os.system(cmd)
