#!/usr/bin/env python
#coding: utf-8

import os
import os.path as osp
from soma import aims
import argparse


def remove_ref_from_headers(fp):
    print fp
    i = aims.read(fp)
    i.header().update({'referentials':[],'transformations':[]})
    s = osp.basename(fp).split('.')
    basename, ext = s[0], '.'.join(s[1:])
    fp2 = osp.join(osp.dirname(fp), '%s_nohdr.%s'%(basename, ext))
    print 'writing', fp2
    aims.write(i, fp2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='`remove_headers.py` is used '\
        'to remove referentials and transformations from the headers of a series'\
        'of files.')
    parser.add_argument('files', nargs='+', help='Files to clean the headers of')
    opts = parser.parse_args()

    print opts.files
    print len(opts.files)

    ans = raw_input('Continue ?')
    for each in opts.files:
        remove_ref_from_headers(each)
