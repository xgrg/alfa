import sys
import nipype.interfaces.spm as spm
import nipype.pipeline.engine as pe
import os, json
from glob import glob
import os.path as osp


wd= '/tmp/t1_dartel'
od = '/tmp/t1_dartel/'

rdwinh = glob(osp.join(wd, 'wr*_nohdr.nii'))
print rdwinh, len(rdwinh)
ans = raw_input('Continue ?')

nodes = []
for each in rdwinh:
    s = osp.split(each)[1].split('_')[0][2:]
    print s
    n = pe.Node(spm.Coregister(), name='Coreg%s'%s)
    n.inputs.target = '/home/grg/data/templates/MNI_atlas_templates/MNI_T1.nii'

    n.inputs.source = each
    nodes.append(n)

w = pe.Workflow(name='RealignMNI')
w.base_dir = od
w.add_nodes(nodes)
w.run('MultiProc', plugin_args={'n_procs' : 6})
