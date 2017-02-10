#! /usr/bin/env python
from glob import glob
import os.path as osp
import os
import nipype.interfaces.spm as spm
import json

subjects = json.load(open('/home/grg/spm/data/subjects_dartel.json'))
wd = '/tmp/dartel/'
c1 = []#glob(osp.join(wd, '*c1.nii'))
c2 = []#glob(osp.join(wd, '*c1.nii'))
for each in subjects:
    print each
    c1.append(glob(osp.join(wd, 'r%s*_c1.nii'%each))[0])
    c2.append(glob(osp.join(wd, 'r%s*_c2.nii'%each))[0])

print c1, c2, len(c1), len(c2)

print 'DARTEL'
dartel = spm.DARTEL()
dartel.inputs.image_files = [c1, c2]
dartel.run()
