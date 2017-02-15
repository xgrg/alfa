import sys
import nipype.interfaces.spm as spm
import nipype.pipeline.engine as pe
import os, json
from glob import glob
import os.path as osp

subjects = json.load(open('/home/grg/spm/data/subjects_dartel.json'))
ff = []
rdwi = []

wd= '/home/grg/dartel_csf0'
subjects2 = []
for s in subjects:
    print s
    try:
        fffp = glob(osp.join(wd, 'u_r*%s*_mabonlm_nobias_spm_c1_Template.nii'%s))[0]
        rdwifp = glob(osp.join('/tmp/dartel', 'r%s*_MD_t1space_wo_csf.nii'%s))[0]
        #rt1fp = glob(osp.join(wd, 'r%s_mabonlm_nobias_spm_c1.nii'%s))[0]
        #t1fp = glob(osp.join(wd, '%s_mabonlm_nobias_spm_c1.nii'%s))[0]
        rdwi.append(rdwifp)
        #rt1.append(rt1fp)
        #t1.append(t1fp)
        ff.append(fffp)
        subjects2.append(s)
    except:
        print s, 'erreur'
print len(rdwi)

print ff, rdwi

ans = raw_input('Continue ?')

nm = pe.Node(spm.DARTELNorm2MNI(), name='Norm2MNI')
nm.inputs.template_file = osp.join(wd, 'Template_6.nii')
nm.inputs.flowfield_files = ff
#nm.inputs.fwhm = [0, 0, 0]
nm.inputs.apply_to_files = rdwi
nm.inputs.modulate = False


w = pe.Workflow(name='DARTELNorm2MNI')
w.base_dir = '/tmp/dartel'
w.add_nodes([nm])
w.run('MultiProc', plugin_args={'n_procs' : 6})
