from string import Template
import subprocess
import os.path as osp
import os
from brainvisa import axon

import argparse
import textwrap

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
              "ALFA Denoised Nobias SPM Scalp"
              ]
    s = []
    for t in types:
        options = {'_database': db.directory}
        w = neuroHierarchy.WriteDiskItem(t, neuroProcesses.getAllFormats())
        fp = w.findValue(dsk)
        s.append(fp)

    directory_path = os.path.dirname(s[0].fullPath())
    batch_location = os.path.join(directory_path, 'spm12_segment_job.m')
    seg8 = osp.splitext(s[0].fullPath())[0].replace('grey', 'bias_corrected') \
                + '_seg8.mat'
    tpm = '/usr/local/MATLAB/R2014a/toolbox/spm12/toolbox/Seg/TPM.nii'
    cmd = 'python -m brainvisa.axon.runprocess --enabledb SPM12Segment_generic'\
        ' TPM_template=%s batch_location=%s seg8_mat=%s t1mri=%s grey_native=%s'\
        ' white_native=%s csf_native=%s skull_native=%s scalp_native=%s'
    cmd = cmd%(tpm, batch_location, seg8, dsk.fullPath(),
                s[0], s[1], s[2], s[3], s[4])
    print cmd
    os.system(cmd)
