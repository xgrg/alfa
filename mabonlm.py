from string import Template
import subprocess
import os.path as osp
import os
import argparse
import textwrap

def createScript(source, text):
    try:
        with open(source, 'w') as f:
            f.write(text)
    except IOError:
        return False
    return True

def parseTemplate(dict, template):
    with open(template, 'r') as f:
        return Template(f.read()).safe_substitute(dict)


def denoise(source):

    assert(osp.isfile(source))
    filename, ext = osp.splitext(source)
    print ext
    assert(ext in ['.nii', '.gz'])
    if ext == '.gz':
       print 'unzipping %s'%source
       os.system('gunzip %s'%source)
       source = filename

    target = '%s_mabonlm.nii'%osp.splitext(source)[0]
    print source, '->', target
    workingDir = osp.split(source)[0]
    tpl_fp = '/home/grg/denoising/mabonlm.tpl'
    matlab_tpl = '/home/grg/denoising/matlab.tpl'

    tags={ 'source': source,
           'target': target,
           'workingDir': workingDir}

    tags={ 'source': source,
           'outputext': 'mabonlm',
           'workingDir': workingDir,
           'beta': 1,
           'rician': 1,
           'nbthreads': 32}

    template = parseTemplate(tags, tpl_fp)

    import tempfile
    code, tmpfile = tempfile.mkstemp(suffix='.m')
    print 'creating tempfile %s'%tmpfile
    createScript(tmpfile, template)

    tmpbase = osp.splitext(tmpfile)[0]
    tags={ 'script': tmpbase, 'workingDir': workingDir}
    cmd = parseTemplate(tags, matlab_tpl)
    print cmd
    os.system(cmd)

    nobias = '%s_nobias.nii'%osp.splitext(target)[0]
    assert(osp.isfile(target))
    cmd = 'N4BiasFieldCorrection -d 3 -i %s -o %s'%(target, nobias)
    print cmd
    os.system(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent('''\
            Denoising using MABONLM
            '''))

    parser.add_argument("-i", dest='input', type=str, required=True)
    args = parser.parse_args()
    source = args.input
    denoise(source)



