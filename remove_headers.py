#!/usr/bin/env python
#coding: utf-8

import os
import os.path as osp
from soma import aims
import argparse


def remove_ref_from_headers(fp, replace=False):
    print fp
    i = aims.read(fp)
    i.header().update({'referentials':[],'transformations':[]})
    s = osp.basename(fp).split('.')
    basename, ext = s[0], '.'.join(s[1:])
    suffix = '_nohdr' if not replace else ''
    fp2 = osp.join(osp.dirname(fp), '%s%s.%s'%(basename, suffix, ext))
    print 'writing', fp2
    aims.write(i, fp2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='`remove_headers.py` is used '\
        'to remove referentials and transformations from the headers of a series'\
        'of files.')
    parser.add_argument('files', nargs='+', help='Files to clean the headers of')
    parser.add_argument('--replace', action='store_true',
        help='replace the files without creating copies (HEADS UP)')
    parser.add_argument('--doit', action='store_true',
        help='do it without prompting')
    opts = parser.parse_args()

    print opts.files
    print len(opts.files)

    if not opts.doit: ans = raw_input('Continue ?')
    for each in opts.files:
        remove_ref_from_headers(each, opts.replace)
