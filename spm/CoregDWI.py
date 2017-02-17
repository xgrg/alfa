import sys
import nipype.interfaces.spm as spm
import nipype.pipeline.engine as pe
import os, json
from glob import glob
import os.path as osp

subjects = json.load(open('/home/grg/spm/data/subjects_dartel.json'))
dwi = []
t1 = []
rt1 = []

wd= '/tmp/t1_dartel'
od = '/tmp/t1_dartel'

subjects2 = []
for s in subjects:
    print s
    try:
        dwifp = glob(osp.join('/home/grg/spm/dartel/T1', '%s*_mabonlm_nobias.nii'%s))[0]
        rt1fp = glob(osp.join(wd, 'r%s_mabonlm_nobias_spm_c1.nii'%s))[0]
        t1fp = glob(osp.join(wd, '%s_mabonlm_nobias_spm_c1.nii'%s))[0]
        dwi.append(dwifp)
        rt1.append(rt1fp)
        t1.append(t1fp)
        subjects2.append(s)
    except:
        print s, 'erreur'

subjects = subjects2
print len(dwi)

ans = raw_input('Continue ?')

nodes = []
for i, s in enumerate(subjects):
    n = pe.Node(spm.Coregister(), name='Coreg%s'%s)
    n.inputs.target = rt1[i]
    n.inputs.source = t1[i]
    n.inputs.apply_to_files = dwi[i]
    nodes.append(n)

w = pe.Workflow(name='RealignT1onC1')
w.base_dir = od
w.add_nodes(nodes)
w.run('MultiProc', plugin_args={'n_procs' : 6})
