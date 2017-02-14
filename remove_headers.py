import sys
import nipype.interfaces.spm as spm
import nipype.pipeline.engine as pe
import os, json
from glob import glob
import os.path as osp
from soma import aims

wd = '/tmp/dartel/'

normalized_files = glob(osp.join(wd, 'DARTELNorm2MNI/Norm2MNI' , 'swr*csf.nii'))

print normalized_files, len(normalized_files)

ans = raw_input('Continue ?')

for each in normalized_files:
    print each
    i = aims.read(each)
    i.header().update({'referentials':[],'transformations':[]})
    fp = osp.join(wd, '%s_nohdr.nii'%osp.split(osp.splitext(each)[0])[1])
    print fp
    aims.write(i, fp)
