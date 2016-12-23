import numpy as np
from nilearn import image
import pandas as pd
from glob import glob
import nibabel as nib
from scipy import stats
from sklearn import linear_model

#=======================================================================
# This is a first draft of how to use linear regression to correct
# for the effect of a parameter given as an image into another image.
#
# This is used for instance to correct potential effect of morphological
# variations in images describing diffusivity, in order to cancel
# potential nuisance from macroscopical changes on the study of
# microstructure.
#======================================================================

# Taking the list of subjects from a previously used design matrix
import json
subjects = json.load(open('/home/grg/spm/data/subjects.json'))

found = 0
notfound = []
data = {}
for subject in subjects:
    #print subject
    if subject in [11703]:
        print 'skipping', subject
        continue
    try:
        md = glob('/home/grg/spm/MD/*%s*.nii'%subject)[0]
        jac = glob('/home/grg/spm/Jacobians/*%s*.nii'%subject)[0]
        found += 1
        data[subject] = (md, jac)
    except IndexError:
        print '####', subject, 'not found'
        notfound.append(subject)

print found, 'over', len(subjects)
print 'not found:', notfound

#============================================

md4d_fp = '/home/grg/spm/MD.nii.gz'
jac4d_fp = '/home/grg/spm/Jac.nii.gz'
print 'Building MD 4D volume'
#md4d = image.concat_imgs([e[0] for s,e in data.items()])
#md4d.to_filename(md4d_fp)

print 'Building Jacobians 4D volume'
#jac4d = image.concat_imgs([e[1] for s,e in data.items()])
#jac4d.to_filename(jac4d_fp)

#============================================

md4d_fp = '/home/grg/spm/MD.nii.gz'
jac4d_fp = '/home/grg/spm/Jac.nii.gz'
jac4dr_fp = '/home/grg/spm/Jac_r.nii.gz'

print 'Resampling Jacobian maps to MD'
# Resampling MD to Jacobian maps resolution
#image.resample_to_img(jac4d_fp, md4d_fp).to_filename(jac4dr_fp)

mdsw_fp = '/home/grg/spm/MD_sw.npy'
jacrsw_fp = '/home/grg/spm/Jac_r_sw.npy'

print 'Swapping MD'
#md = nib.load(md4d_fp).dataobj
#mdsw = np.swapaxes(md, 0, 3)
#np.save(mdsw_fp, mdsw)
#mdsw = np.load(mdsw_fp)

print 'Swapping Jac'
#jacr = nib.load(jac4dr_fp).dataobj
#jacrsw = np.swapaxes(jacr, 0, 3)
#np.save(jacrsw_fp, jacrsw)
#jacrsw = np.load(jacrsw_fp)

#nb_subjects, dimX, dimY, dimZ = mdsw.shape
#print 'Iterating linear model', mdsw.shape, jacrsw.shape
#r_list = []
#err_list = []
#ypred_list = []
#ycorr_list = []


#for i in xrange(dimX):
#    print i
#    for j in xrange(dimY):
#        for k in xrange(dimZ):
#            y,x = mdsw[:,i,j,k], jacrsw[:,i,j,k]
#
#            slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
#
#            #err_list.append(std_err)
#            r_list.append(r_value)
#            ypred = x*slope + intercept
#            err = ypred - y
#            ycorr = np.mean(y) - err
#
#            #ypred_list.extend(ypred)
#            ycorr_list.extend(ycorr)
#            #err_list.extend(err)

print 'saving r list'
#np.save('/tmp/r_list.npy', np.array(r_list))

#err = np.array(err_list).reshape(mdsw.shape)
#ypred = np.array(ypred_list).reshape(mdsw.shape)
#ycorr = np.array(ycorr_list).reshape(mdsw.shape)

print 'saving errors'
#np.save('/tmp/err.npy', np.array(err))
print 'saving y pred'
#np.save('/tmp/ypred.npy', np.array(ypred))
print 'saving y corr'
#np.save('/tmp/ycorr.npy', ycorr)

#ima = image.new_img_like('/home/grg/spm/MD/10134_MD_MNIspace_s.nii', pmap_im)
#ima.to_filename('/tmp/p_map2.nii.gz')
#
#ima = image.new_img_like('/home/grg/spm/MD/10134_MD_MNIspace_s.nii', stderr_im)
#ima.to_filename('/tmp/stderr_map2.nii.gz')

print 'applying correction to every map'
#
ypred = np.load('/tmp/ypred.npy').reshape((109,91,91, 514))
ycorr = np.load('/tmp/ycorr.npy').reshape((109,91,91, 514))

print 'first building r-value map'
#r = np.load('/tmp/r_list.npy')
#r = np.swapaxes(r.reshape(ypred.shape[:]), 1, 2)
#r = np.swapaxes(r, 0 ,1)
#r_im = image.new_img_like('/home/grg/spm/MD/10134_MD_MNIspace_s.nii', r)
#r_im.to_filename('/tmp/r_map.nii.gz')

print ypred.shape, ycorr.shape#, r.shape

for i, subject in enumerate(subjects):
    print subject, i
    try:
        mdfp = glob('/home/grg/spm/MD/*%s*.nii'%subject)[0]
        md = np.array(nib.load(mdfp).dataobj)
        print 'md', md.shape
        print ycorr.shape
        md_corr = ycorr[:,:,:,i].reshape(ycorr.shape[:3]) #(dimX, dimZ, dimY))
        md_corr = np.swapaxes(md_corr, 1,2)
        md_corr = np.swapaxes(md_corr, 0,1)
        print md_corr.shape
        img = image.new_img_like(mdfp, md_corr)
        img.to_filename('/home/grg/spm/MD_corr/%s_MD_corr.nii'%subject)
        md_pred = ypred[:,:,:,i].reshape(ypred.shape[:3])
        md_pred = np.swapaxes(md_pred, 1,2)
        md_pred = np.swapaxes(md_pred, 0,1)
        img = image.new_img_like(mdfp, md_pred)
        img.to_filename('/home/grg/spm/MD_pred/%s_MD_pred.nii'%subject)

    except IndexError:
        print subject, 'missing'
