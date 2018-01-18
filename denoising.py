from string import Template
import subprocess
import os.path as osp
import os
import logging as log
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


def launchCommand(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=None, nice=0):
    import sys
    sys.path.append('/home/grg/toad')
    from lib import util
    """Execute a program in a new process

    Args:
	command: a string representing a unix command to execute
	stdout: this attribute is a file object that provides output from the child process
	stderr: this attribute is a file object that provides error from the child process
	timeout: Number of seconds before a process is consider inactive, usefull against deadlock
	nice: run cmd  with  an  adjusted  niceness, which affects process scheduling

    Returns
	return a 3 elements tuples representing the command execute, the standards output and the standard error message

    Raises
	OSError:      the function trying to execute a non-existent file.
	ValueError :  the command line is called with invalid arguments

    """
    binary = cmd.split(" ").pop(0)
    if util.which(binary) is None:
	print ("Command {} not found".format(binary))

    print ("Launch {} command line...".format(binary))
    print ("Command line submit: {}".format(cmd))

    (executedCmd, output, error)= util.launchCommand(cmd, stdout, stderr, timeout, nice)
    if not (output is "" or output is "None" or output is None):
	print("Output produce by {}: {} \n".format(binary, output))

    if not (error is '' or error is "None" or error is None):
	print("Error produce by {}: {}\n".format(binary, error))


def rescale(source, target):
    import nibabel as nib
    import numpy as np
    from nilearn import image

    n = nib.load(source)
    d = np.array(n.dataobj)
    s = n.dataobj.slope
    i = image.new_img_like(n, d/s)
    i.to_filename(target)
    log.info('Rescaling done: %s rescaled to %s'%(source, target))

def denoise(source, do_rescale=False):
    import tempfile
    assert(osp.isfile(source))
    filename, ext = osp.splitext(source)
    assert(ext in ['.nii', '.gz'])

    tpl_fp = '/home/grg/denoising/denoise.tpl'
    matlab_tpl = '/home/grg/denoising/matlab.tpl'

    if ext == '.gz' and not rescale:
       log.info('unzipping %s'%source)
       os.system('gunzip %s'%source)
       source = filename

    target = '%s_denoised.nii'%osp.splitext(source)[0]
    log.info('%s -> %s'%(source, target))

    if do_rescale:
        log.info('Rescaling...')
        rescaled_fp = target.replace('denoised', 'rescaled')
        rescale(source, rescaled_fp)
        source = rescaled_fp

    workingDir = osp.split(source)[0]

    tags={ 'source': source,
           'target': target,
           'workingDir': workingDir,
           'beta': 1,
           'rician': 1,
           'nbthreads': 32}

    template = parseTemplate(tags, tpl_fp)

    code, tmpfile = tempfile.mkstemp(suffix='.m')
    log.info('creating tempfile %s'%tmpfile)
    createScript(tmpfile, template)

    tmpbase = osp.splitext(tmpfile)[0]
    tags={ 'script': tmpbase, 'workingDir': workingDir}
    cmd = parseTemplate(tags, matlab_tpl)
    log.info(cmd)
    os.system(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent('''\
            Denoising using LPCA
            '''))

    parser.add_argument('input', type=str)
    parser.add_argument('--rescale', action='store_true')
    args = parser.parse_args()
    log.basicConfig(level=log.INFO)
    source = args.input
    denoise(source, args.rescale)
