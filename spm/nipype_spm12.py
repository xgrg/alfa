from string import Template
import subprocess
import os.path as osp
import os
from brainvisa import axon

import argparse
import textwrap

def nipype_spm12(tpm, source, s):
    import nipype.interfaces.spm as spm
    seg = spm.NewSegment()
    seg.inputs.channel_files = source
    tpm_path = '/usr/local/MATLAB/R2014a/toolbox/spm12/toolbox/Seg/TPM.nii'
    tissue1 = ((tpm, 1), 2, (True,True), (False, False))
    tissue2 = ((tpm, 2), 2, (True,True), (False, False))
    tissue3 = ((tpm, 3), 2, (True,False), (False, False))
    tissue4 = ((tpm, 4), 2, (True,False), (False, False))
    tissue5 = ((tpm, 5), 2, (True,False), (False, False))
    seg.inputs.tissues = [tissue1, tissue2, tissue3, tissue4, tissue5]
    res = seg.run()
    for fp1, fp2 in zip(res.outputs.native_class_images, s[:5]):
        cmd = 'mv %s %s'%(fp1[0], fp2)
        print(cmd)
        os.system(cmd)
    for fp1, fp2 in zip(res.outputs.dartel_input_images, s[5:]):
        cmd = 'mv %s %s'%(fp1[0], fp2)
        print(cmd)
        os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent('''\
            SPM12
            '''))

    parser.add_argument("-i", dest='input', type=str, required=True)
    args = parser.parse_args()
    axon.initializeProcesses()
    import neuroHierarchy, neuroProcesses
    source = args.input
    dsk = neuroHierarchy.databases.createDiskItemFromFileName(source)
    db = neuroHierarchy.databases._databases[dsk.get('_database')]

    types = ["ALFA Denoised Nobias SPM Grey matter",
              "ALFA Denoised Nobias SPM White matter",
              "ALFA Denoised Nobias SPM CSF",
              "ALFA Denoised Nobias SPM Skull",
              "ALFA Denoised Nobias SPM Scalp",
              "ALFA Denoised Nobias SPM DARTEL-imported Grey matter",
              "ALFA Denoised Nobias SPM DARTEL-imported White matter"
              ]
    s = []
    for t in types:
        options = {'_database': db.directory}
        w = neuroHierarchy.WriteDiskItem(t, neuroProcesses.getAllFormats())
        fp = w.findValue(dsk)
        s.append(fp)

    directory_path = os.path.dirname(s[0].fullPath())

    tpm = '/usr/local/MATLAB/R2014a/toolbox/spm12/toolbox/Seg/TPM.nii'
    cmd = ' TPM_template=%s t1mri=%s grey_native=%s'\
        ' white_native=%s csf_native=%s skull_native=%s scalp_native=%s'
    cmd = cmd%(tpm, dsk.fullPath(),
                s[0], s[1], s[2], s[3], s[4])
    print cmd
    print s
    nipype_spm12(tpm, dsk.fullPath(), s)
