from nilearn import plotting
import nilearn
import os.path as osp
from glob import glob
import matplotlib
from nistats.thresholding import map_threshold
from matplotlib import pyplot as plt
from IPython.display import display_html
plt.rcParams.update({'figure.max_open_warning': 0})
import nibabel as nib
import numpy as np
from PIL import Image
import io
import gzip, pickle
import sys
import pandas as pd
sys.path.append('/home/grg/git/alfa/')
sys.path.append('/home/grg/git/pyAAL/')
import multireg_spm12 as mreg
import pyAAL

def plot_stat_map2(**kwargs):
    cut_coords = kwargs['cut_coords']
    row_l = kwargs['row_l']
    lines_nb = int(len(cut_coords) / row_l)
    for line in xrange(lines_nb):
        opt = dict(kwargs)
        opt.pop('row_l')
        opt['cut_coords'] = cut_coords[line * row_l: (line +1) *row_l]
        plotting.plot_stat_map(**opt)



def plot_stat_map(img, start, end, step=1, row_l=6, title='', bg_img=None,
    threshold=None, axis='z', method='plot_stat_map', overlay=None, pngfile=None):
    ''' Generates a multiple row plot instead of the very large native plot,
    given the number of slices on each row, the index of the first/last slice
    and the increment.

    Method parameter can be plot_stat_map (default) or plot_prob_atlas.'''

    slice_nb = int(abs(((end - start) / float(step))))
    images = []
    for line in range(int(slice_nb/float(row_l) + 1)):
        opt = {'title':{True:title,
                        False:None}[line==0],
               'colorbar':True,
               'black_bg':True,
               'display_mode':axis,
               'threshold':threshold,
               'cut_coords':range(start + line * row_l * step,
                                       start + (line+1) * row_l * step,
                                       step)}
        if method == 'plot_prob_atlas':
            opt.update({'maps_img': img,
                        'view_type': 'contours'})
        elif method == 'plot_stat_map':
            opt.update({'stat_map_img': img})
        if not bg_img is None:
            opt.update({'bg_img': bg_img})

        t = getattr(plotting, method).__call__(**opt)

        # Add overlay
        if not overlay is None:
            if isinstance(overlay, list):
                for each in overlay:
                    t.add_overlay(each)
            else:
                t.add_overlay(overlay)

        # Converting to PIL and appending it to the list
        buf = io.BytesIO()
        t.savefig(buf)
        buf.seek(0)
        im = Image.open(buf)
        images.append(im)

    # Joining the images
    imsize = images[0].size
    out = Image.new('RGBA', size=(imsize[0], len(images)*imsize[1]))
    for i, im in enumerate(images):
        box = (0, i * imsize[1], imsize[0], (i+1) * imsize[1])
        out.paste(im, box)

    if pngfile is None:
        import tempfile
        pngfile = tempfile.mkstemp(suffix='.png')[1]
    print 'Saving to...', pngfile

    out.save(pngfile)
    return pngfile


def plot_two_maps(img, overlay, start, end, row_l=6, step=1, title='',
    threshold=None, axis='z', method='plot_stat_map', pngfile=None):
    ''' Similar to plot_stat_map, generates a multiple row plot instead of the
    very large native plot, given the number of slices on each row, the index of
    the first/last slice and the increment.

    This function is used to compare two maps (segmentation, statistical
    clusters, ...).'''

    slice_nb = int(abs(((end - start) / float(step))))
    images = []
    for line in range(int(slice_nb/float(row_l) + 1)):
        opt = {'title':{True:title,
                        False:None}[line==0],
               'colorbar':True,
               'black_bg':True,
               'display_mode':axis,
               'threshold':threshold,
               'cmap': plotting.cm.blue_transparent,
               'cut_coords':range(start + line * row_l * step,
                                       start + (line+1) * row_l * step,
                                       step)}
        if method == 'plot_prob_atlas':
            opt.update({'maps_img': img,
                        'view_type': 'contours'})
        elif method == 'plot_stat_map':
            opt.update({'stat_map_img': img})

        t = getattr(plotting, method).__call__(**opt)

        # Add overlay
        if not overlay is None:
            if isinstance(overlay, list):
                for each in overlay:
                    t.add_overlay(each, cmap=plotting.cm.red_transparent)
            else:
                t.add_overlay(overlay, cmap=plotting.cm.red_transparent)

        # Converting to PIL and appending it to the list
        buf = io.BytesIO()
        t.savefig(buf)
        buf.seek(0)
        im = Image.open(buf)
        images.append(im)

    # Joining the images
    imsize = images[0].size
    out = Image.new('RGBA', size=(imsize[0], len(images)*imsize[1]))
    for i, im in enumerate(images):
        box = (0, i * imsize[1], imsize[0], (i+1) * imsize[1])
        out.paste(im, box)

    if pngfile is None:
        import tempfile
        pngfile = tempfile.mkstemp(suffix='.png')[1]
    print 'Saving to...', pngfile, '(%s)'%title

    out.save(pngfile)


def glassbrain_allcontrasts(path, title, mode='uncorrected',
    cluster_threshold=50):
    ''' For each SPM contrast from a Nipype workflow (`path` points to the base
    directory), generates a glass brain figure with the corresponding
    thresholded map.

    `mode` can be either 'uncorrected' (p<0.001, T>3.1, F>4.69)
                      or 'FWE' (p<0.05, T>4.54, F>8.11).
    `title` is the title displayed on the plot.'''

    nodes = [pickle.load(gzip.open(osp.join(path, e, '_node.pklz'), 'rb'))
        for e in ['modeldesign', 'estimatemodel','estimatecontrasts']]
    _, _, node = nodes

    spm_mat_file = osp.join(node.output_dir(), 'SPM.mat')
    for i in range(1, len(node.inputs.contrasts)+1):
        img = glob(osp.join(node.output_dir(), 'spm*_00%02d.nii'%i))[0]
        contrast_type = osp.split(img)[-1][3]
        print img, contrast_type
        contrast_name = node.inputs.contrasts[i-1][0]

        thresholded_map1, threshold1 = map_threshold(img, threshold=0.001,
            cluster_threshold=cluster_threshold)
        if mode == 'uncorrected':
            threshold1 = 3.106880 if contrast_type == 'T' else 4.69
            pval_thresh = 0.001
        elif mode == 'FWE':
            threshold1 = 4.5429 if contrast_type == 'T' else 8.1101
            pval_thresh = 0.05

        plotting.plot_glass_brain(thresholded_map1, colorbar=True, black_bg=True,
            display_mode='ortho', threshold=threshold1,
            title='(%s) %s - %s>%.02f - p<%s (%s)'
            %(title, contrast_name, contrast_type, threshold1, pval_thresh,
            mode))


def sections_allcontrasts(path, title, contrasts='all', mode='uncorrected',
    axis='z', cluster_threshold=50, row_l=8, start=-32, end=34, step=2):
    ''' For each SPM contrast from a Nipype workflow (`path` points to the base
    directory), generates a figure made of slices from the corresponding
    thresholded map.

    `mode` can be either 'uncorrected' (p<0.001, T>3.1, F>4.69)
                      or 'FWE' (p<0.05, T>4.54, F>8.11).
    `title` is the title displayed on the plot.'''

    nodes = [pickle.load(gzip.open(osp.join(path, e, '_node.pklz'), 'rb'))
        for e in ['modeldesign', 'estimatemodel','estimatecontrasts']]
    _, _, node = nodes

    def _thiscontrast(i, node=node, cluster_threshold=cluster_threshold, mode=mode,
            title=title, axis=axis, row_l=row_l, start=start, end=end, step=step):
        img = glob(osp.join(node.output_dir(), 'spm*_00%02d.nii'%i))[0]
        contrast_type = osp.split(img)[-1][3]
        print img, contrast_type
        contrast_name = node.inputs.contrasts[i-1][0]
        thresholded_map1, threshold1 = map_threshold(img, threshold=0.001,
            cluster_threshold=cluster_threshold)
        if mode == 'uncorrected':
            threshold1 = 3.106880 if contrast_type == 'T' else 4.69
            pval_thresh = 0.001
        elif mode == 'FWE':
            threshold1 = 4.5429 if contrast_type == 'T' else 8.1101
            pval_thresh = 0.05

        pngfile = plot_stat_map(img, threshold=threshold1, row_l=row_l, axis=axis,
            start=start, end=end, step=step, title= '(%s) %s - %s>%.02f - p<%s (%s)'
            %(title, contrast_name, contrast_type, threshold1, pval_thresh,
            mode))
        return pngfile

    sections = []
    for i in range(1, len(node.inputs.contrasts)+1):
        if contrasts == 'all' or i in contrasts:
            pngfile = _thiscontrast(i)
            sections.append((node.inputs.contrasts[i-1][0], pngfile))
    return sections
