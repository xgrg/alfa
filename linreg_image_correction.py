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
df = pd.read_excel('/home/grg/spm/designmatrix2.xls')
subjects = df['Subj_ID'].tolist()

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
#pval_list = []
#stderr_list = []

method = 'normal'
print method

#for i in xrange(dimX):
#    print i
#    for j in xrange(dimY):
#        for k in xrange(dimZ):
#            y,x = mdsw[:,i,j,k], jacrsw[:,i,j,k]
#
#
#            if method == 'normal':
#                slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
#
#            elif method == 'ransac':
#                x = x.reshape((x.size, 1))
#                y = y.reshape((y.size, 1))
#                model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
#                model_ransac.fit(x, y)
#                sse = np.sum((model_ransac.predict(x) - y) ** 2, axis=0) / float(x.shape[0] - x.shape[1])
#                se = np.array([
#                        np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(x.T, x))))
#                                                                    for i in range(sse.shape[0])
#			])
#                t = model_ransac.estimator_.coef_ / se
#                p_value = 2 * (1 - stats.t.cdf(np.abs(t), y.shape[0] - x.shape[1]))
#                std_err = se
#
#            stderr_list.append(std_err)
#            pval_list.append(p_value)

print 'swapping and saving'
#np.save('/tmp/pval_list.npy', np.array(pval_list))
#np.save('/tmp/stderr_list.npy', np.array(stderr_list))
#pmap_im = np.array(pval_list).reshape((dimX, dimY, dimZ))
#stderr_im = np.array(stderr_list).reshape((dimX, dimY, dimZ))
#pmap_im = np.swapaxes(pmap_im, 1,2)
#stderr_im = np.swapaxes(stderr_im, 1,2)

#ima = image.new_img_like('/home/grg/spm/MD/10134_MD_MNIspace_s.nii', pmap_im)
#ima.to_filename('/tmp/p_map2.nii.gz')
#
#ima = image.new_img_like('/home/grg/spm/MD/10134_MD_MNIspace_s.nii', stderr_im)
#ima.to_filename('/tmp/stderr_map2.nii.gz')

print 'applying correction to every map'

stderr_im = np.array(nib.load('/tmp/stderr_map2.nii.gz').dataobj)
stderr_im = np.swapaxes(stderr_im, 0, 1)

for i, subject in enumerate(subjects):
    subject = str(subject)[:5]
    print subject, i
    try:
        mdfp = glob('/home/grg/spm/MD/*%s*.nii'%subject)[0]
        md = np.array(nib.load(mdfp).dataobj)
        md_corr = md + stderr_im
        img = image.new_img_like(mdfp, md_corr)
        img.to_filename('/home/grg/spm/MD_corr/%s_MD_corr.nii'%subject)

    except IndexError:
        print subject, 'missing'
