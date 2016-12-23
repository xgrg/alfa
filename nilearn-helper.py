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

def plot_stat_map2(**kwargs):
    cut_coords = kwargs['cut_coords']
    row_l = kwargs['row_l']
    lines_nb = int(len(cut_coords) / row_l)
    for line in xrange(lines_nb):
        opt = dict(kwargs)
        opt.pop('row_l')
        opt['cut_coords'] = cut_coords[line * row_l: (line +1) *row_l]
        plotting.plot_stat_map(**opt)



def plot_stat_map(img, start, end, step=1, row_l=6, title='', threshold=None,
                  axis='z', method='plot_stat_map', overlay=None, pngfile=None):
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
               #'bg_img': bg_img,
               'threshold':threshold,
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


def plot_two_maps(img, overlay, start, end, row_l=6, step=1, title='', threshold=None,
                  axis='z', method='plot_stat_map', pngfile=None):
    ''' Similar to plot_stat_map, generates a multiple row plot instead of the very
    large native plot, given the number of slices on each row, the index of the first/last slice
    and the increment.

    This function is used to compare two maps (segmentation, statistical clusters, ...).'''
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
    print 'Saving to...', pngfile

    out.save(pngfile)
