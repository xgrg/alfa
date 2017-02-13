import sys
import nipype.interfaces.spm as spm
import nipype.pipeline.engine as pe
import os, json
from glob import glob
import os.path as osp
import numpy as np
import nibabel as nib
from nilearn import image

subjects = json.load(open('/home/grg/spm/data/subjects_dartel.json'))


for s in subjects:
  try:
    print s
    mdfp = [e for e in glob('/home/grg/dartel/%s*_t1space.nii*'%s) if not '.minf' in e][0]
    md = np.asarray(nib.load(mdfp).dataobj)
    for i in [3,4,5]:
       csffp = glob('/home/grg/data/ALFA_DWI/%s*/T1/%s*c%s*.nii'%(s,s,i))[0]
       print csffp
       csf = np.asarray(nib.load(csffp).dataobj)
       md[csf>0.5] = 0
    image.new_img_like(mdfp, md).to_filename('/tmp/dartel5/%s_MD_t1space_wo_csf.nii'%s)
  except IndexError:
    print s, 'error'

