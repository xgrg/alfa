#!/usr/bin/env python
import argparse
import textwrap
try:
    import nipype
    from builtins import range
    import nipype.interfaces.io as nio           # Data i/o
    import nipype.interfaces.spm as spm          # spm
    import nipype.interfaces.fsl as fsl          # fsl
    import nipype.interfaces.matlab as mlab      # how to run matlab
    import nipype.interfaces.fsl as fsl          # fsl
    import nipype.interfaces.utility as util     # utility
    import nipype.pipeline.engine as pe          # pypeline engine
    import nipype.algorithms.modelgen as model   # model specification
    from nipype.interfaces.matlab import MatlabCommand
    from nipype.interfaces import spm
    from nipype.interfaces.spm.model import MultipleRegressionDesign, FullFactorialDesign
    import pandas as pd
    import os.path as osp
    import os
    from glob import glob

except ImportError as e:
    raise ImportError('Did you activate jupyter virtualenv (nipype) ?')

MatlabCommand.set_default_paths('/usr/local/MATLAB/R2014a/toolbox/spm12')
MatlabCommand.set_default_matlab_cmd('matlab -nodesktop -nosplash')

# === Building contrasts vectors depending on the analysis ===

def make_contrasts_interaction_linearage():
    cont1 = ('Apo2-3>Apo2-4', 'T', ['Apoe2-3', 'Apoe2-4'], [1,-1])
    cont2 = ('Apo2-4>Apo3-3', 'T', ['Apoe2-4', 'Apoe3-3'], [1,-1])
    cont3 = ('Apo3-3>Apo3-4', 'T', ['Apoe3-3', 'Apoe3-4'], [1,-1])
    cont4 = ('Apo3-4>Apo4-4', 'T', ['Apoe3-4', 'Apoe4-4'], [1,-1])
    cont5 = ('Main effect ApoE', 'F', [cont1, cont2, cont3, cont4])
    cont6 = ('age23>age24', 'T', ['age23', 'age24'], [1,-1])
    cont7 = ('age24>age33', 'T', ['age24', 'age33'], [1,-1])
    cont8 = ('age33>age34', 'T', ['age33', 'age34'], [1,-1])
    cont9 = ('age34>age4-4', 'T', ['age34', 'age44'], [1,-1])
    cont10 = ('Interaction linear age-genotype', 'F', [cont6, cont7, cont8, cont9])
    cont11 = ('C<NC', 'T', ['Apoe2-3', 'Apoe2-4', 'Apoe3-3', 'Apoe3-4', 'Apoe4-4'], [3, -2, 3, -2, -2])
    cont12 = ('C>NC', 'T', ['Apoe2-3', 'Apoe2-4', 'Apoe3-3', 'Apoe3-4', 'Apoe4-4'], [-3, 2, -3, 2, 2])
    cont13 = ('HO<HZ', 'T', ['Apoe2-3', 'Apoe2-4', 'Apoe3-3', 'Apoe3-4', 'Apoe4-4'], [1, 1, 1, 1, -4])
    cont14 = ('HO>HZ', 'T', ['Apoe2-3', 'Apoe2-4', 'Apoe3-3', 'Apoe3-4', 'Apoe4-4'], [-1, -1, -1, -1, 4])
    cont15 = ('HO<All_Age_Linear', 'T', ['age23', 'age24', 'age33', 'age34', 'age44'], [3, -2, 3, -2, -2])
    cont16 = ('HO>All_Age_Linear', 'T', ['age23', 'age24', 'age33', 'age34', 'age44'], [-3, 2, -3, 2, 2])
    contrasts = [cont1, cont2, cont3, cont4, cont5, cont6, cont7, cont8, cont9, cont10, cont11, cont12, cont13, cont14, cont15, cont16]
    return contrasts


def make_contrasts(names):

    contrasts = []

    if 'Apoe2-3' in names:
        print 'ApoE groups detected'

        cont1 = ('Apo2-3>Apo2-4', 'T', ['Apoe2-3', 'Apoe2-4'], [1,-1])
        cont2 = ('Apo2-4>Apo3-3', 'T', ['Apoe2-4', 'Apoe3-3'], [1,-1])
        cont3 = ('Apo3-3>Apo3-4', 'T', ['Apoe3-3', 'Apoe3-4'], [1,-1])
        cont4 = ('Apo3-4>Apo4-4', 'T', ['Apoe3-4', 'Apoe4-4'], [1,-1])
        cont5 = ('Main effect ApoE', 'F', [cont1, cont2, cont3, cont4])
        cont6 = ('C<NC', 'T', ['Apoe2-3', 'Apoe2-4', 'Apoe3-3', 'Apoe3-4', 'Apoe4-4'], [3, -2, 3, -2, -2])
        cont7 = ('C>NC', 'T', ['Apoe2-3', 'Apoe2-4', 'Apoe3-3', 'Apoe3-4', 'Apoe4-4'], [-3, 2, -3, 2, 2])
        cont8 = ('HO<HZ', 'T', ['Apoe2-3', 'Apoe2-4', 'Apoe3-3', 'Apoe3-4', 'Apoe4-4'], [1, 1, 1, 1, -4])
        cont9 = ('HO>HZ', 'T', ['Apoe2-3', 'Apoe2-4', 'Apoe3-3', 'Apoe3-4', 'Apoe4-4'], [-1, -1, -1, -1, 4])

        contrasts.extend([cont1, cont2, cont3, cont4, cont5, cont6, cont7, cont8, cont9])

    if 'age23' in names:
        print 'Interaction linear age-genotype detected'

        cont10 = ('age23>age24', 'T', ['age23', 'age24'], [1,-1])
        cont11 = ('age24>age33', 'T', ['age24', 'age33'], [1,-1])
        cont12 = ('age33>age34', 'T', ['age33', 'age34'], [1,-1])
        cont13 = ('age34>age44', 'T', ['age34', 'age44'], [1,-1])
        cont14 = ('Interaction linear age-genotype', 'F', [cont10, cont11, cont12, cont13])
        cont15 = ('C<NC_Age_Linear', 'T', ['age23', 'age24', 'age33', 'age34', 'age44'], [3, -2, 3, -2, -2])
        cont16 = ('C>NC_Age_Linear', 'T', ['age23', 'age24', 'age33', 'age34', 'age44'], [-3, 2, -3, 2, 2])
        cont17 = ('HO<HZ_Age_Linear', 'T', ['age23', 'age24', 'age33', 'age34', 'age44'], [1, 1, 1, 1, -4])
        cont18 = ('HO>HZ_Age_Linear', 'T', ['age23', 'age24', 'age33', 'age34', 'age44'], [-1, -1, -1, -1, 4])

        contrasts.extend([cont10, cont11, cont12, cont13, cont14, cont15, cont16, cont17, cont18])

    if 'agesq23' in names:
        print 'Interaction quadratic age-genotype detected'

        cont19 = ('agesq23>agesq24', 'T', ['agesq23', 'agesq24'], [1,-1])
        cont20 = ('agesq24>agesq33', 'T', ['agesq24', 'agesq33'], [1,-1])
        cont21 = ('agesq33>agesq34', 'T', ['agesq33', 'agesq34'], [1,-1])
        cont22 = ('agesq34>agesq44', 'T', ['agesq34', 'agesq44'], [1,-1])
        cont23 = ('Interaction Age square-genotype', 'F', [cont19, cont20, cont21, cont22])
        cont24 = ('C<NC_Age_Square', 'T', ['agesq23', 'agesq24', 'agesq33', 'agesq34', 'agesq44'], [3, -2, 3, -2, -2])
        cont25 = ('C>NC_Age_Square', 'T', ['agesq23', 'agesq24', 'agesq33', 'agesq34', 'agesq44'], [-3, 2, -3, 2, 2])
        cont26 = ('HO<HZ_Age_Square', 'T', ['agesq23', 'agesq24', 'agesq33', 'agesq34', 'agesq44'], [1, 1, 1, 1, -4])
        cont27 = ('HO>HZ_Age_Square', 'T', ['agesq23', 'agesq24', 'agesq33', 'agesq34', 'agesq44'], [-1, -1, -1, -1, 4])

        contrasts.extend([cont19, cont20, cont21, cont22, cont23, cont24, cont25, cont26, cont27])

    cov = {'age': 'Linear age',
            'agesq': 'Age square',
            'educyears': 'Educational Years',
            'gender': 'Gender'}

    for k, v in cov.items():
        if k in names:
            print 'Main effect of %s'%v
            c = ('Effect %s'%v, 'T', [k], [1])
            contrasts.append(c)

    return contrasts

def make_contrasts_justage_educyears(names):
    if len(names)==7:
        cont1 = ('Effect Age', 'T', ['age'], [1])
        cont2 = ('Effect Gender', 'T', ['gender'], [1])
        contrasts = [cont1, cont2]
        return contrasts
    elif len(names)>7:
        cont1 = ('Effect Age', 'T', ['age'], [1])
        cont2 = ('Effect Gender', 'T', ['gender'], [1])
        cont3 = ('Effect Educational Years', 'T', ['educyears'], [1])
        contrasts = [cont1, cont2, cont3]
    return contrasts

def make_contrasts_agesq(names):
    cont1 = ('Age - Agesq', 'T', ['age','agesq'], [1,-1])
    cont2 = ('Agesq - Age', 'T', ['age','agesq'], [-1,1])
    cont3 = ('Effect Age', 'T', ['age'], [1])
    cont4 = ('Effect Agesq', 'T', ['agesq'], [1])
    contF = ('Main effect Age', 'F', [cont1, cont2])
    contF2 = ('Main effect Age (omnibus)', 'F', [cont3, cont4])
    contrasts = [cont1, cont2, contF, cont3, cont4, contF2]

    return contrasts

def make_contrasts_fullfactorial_version():
    cont1 = ('Positive effect Genotype1', 'T', ['genotype_{1}', 'genotype_{2}'], [1,-1])
    cont2 = ('Positive effect Genotype2', 'T', ['genotype_{2}', 'genotype_{3}'], [1,-1])
    cont3 = ('Positive effect Genotype3', 'T', ['genotype_{3}', 'genotype_{4}'], [1,-1])
    cont4 = ('Positive effect Genotype4', 'T', ['genotype_{4}', 'genotype_{5}'], [1,-1])
    cont6 = ('C<NC', 'T', ['genotype_{1}', 'genotype_{2}', 'genotype_{3}', 'genotype_{4}', 'genotype_{5}'], [3,-2,3,-2,-2])
    cont5 = ('Main effect Genotype', 'F', [cont1, cont2, cont3, cont4])
    contrasts = [cont1, cont2, cont3, cont4, cont5, cont6]
    return contrasts


# === Performing the SPM analysis using the previously prepared regressors (vectors and names), contrasts vectors ... ===

def multiple_regression_analysis(scans, vectors, names, contrasts, destdir, explicitmask, analysis_name='analysis', verbose=True):
    ''' Runs a Multiple Regression analysis over a given type of parametric maps (param),
    using data from an Excel sheet as regressors (columns in 'names')
    and a given explicit mask.

    The whole analysis will be performed in the directory 'destdir'.'''

    print 'Analysis name:', analysis_name

    centering = [1] * len(names)
    if verbose:
        print 'Scans (%s):'%len(scans), scans
        print 'Vectors (%s)'%len(vectors)
        print 'Names (%s):'%len(names), names
        print 'Contrasts (%s):'%len(contrasts), contrasts
    covariates = []
    for name, v, c in zip(names, vectors, centering):
        covariates.append(dict(name=name, centering=c, vector=v))

    model = MultipleRegressionDesign(in_files = scans,
                                    user_covariates = covariates,
                                    explicit_mask_file = explicitmask)

    # Model Estimation
    est = spm.EstimateModel(estimation_method = {'Classical': 1})

    # Contrast Estimation
    con = spm.EstimateContrast(contrasts = contrasts,
                               group_contrast = True)

    # Creating Workflow
    a = pe.Workflow(name=analysis_name)
    a.base_dir = destdir

    n1 = pe.Node(model, name='modeldesign')
    n2 = pe.Node(est, name='estimatemodel')
    n3 = pe.Node(con, name='estimatecontrasts')

    a.connect([(n1, n2, [('spm_mat_file','spm_mat_file')] ),
               (n2,n3, [('spm_mat_file', 'spm_mat_file'),
                        ('beta_images', 'beta_images'),
                        ('residual_image', 'residual_image')]), ])
    a.config['execution']['stop_on_first_rerun'] = True
    return a


# === Versions of the analysis

def generic_version(excel_file, destdir, explicitmask, analysis_name):

    print 'Analysis name:', analysis_name

    data = pd.read_excel(excel_file)

    param = data.columns[0]
    print 'First column:', param
    scans = data[param].tolist()

    names = list(data.columns[1:])
    print 'Columns in the model:', names
    vectors = [data[e].tolist() for e in names]

#    # Defining contrasts based on model..
#    if 'justage' in analysis_name:
#        print '### Justage model identified ###'
#        contrasts = make_contrasts_justage_educyears(names)
#    elif 'educyears' in analysis_name:
#        print '### Linearage and educyears model identified ###'
#        contrasts = make_contrasts_justage_educyears(names)
#    elif 'interaction' in analysis_name:
#        print '### Interaction linearage model identified ###'
#        contrasts = make_contrasts_interaction_linearage()
#    elif 'agesq' in analysis_name:
#        print '### Agesq identified'
#        contrasts = make_contrasts_agesq(names)
#    else:
#        print '### No model type identified: using standard contrasts###'
#        contrasts = make_contrasts_original_version()

    contrasts = make_contrasts(names)

    a = multiple_regression_analysis(scans, vectors, names, contrasts, destdir, explicitmask, analysis_name)
    return a

def fullfactorial(excel_file, destdir, explicitmask, analysis_name):

    print 'Analysis name:', analysis_name

    data = pd.read_excel(excel_file)
    param = data.columns[0]
    print 'First column:', param
    scans = data[param].tolist()

    names = list(data.columns[1:])
    print 'Columns in the model:', names
    vectors = [data[e].tolist() for e in names]

    contrasts = make_contrasts_fullfactorial_version()

    import numpy as np
    groups = [102, 44, 143, 160, 65]
    print sum(groups), len(data.index)
    assert(sum(groups) == len(data.index))
    groups_names = ['Apoe2-3', 'Apoe2-4', 'Apoe3-3', 'Apoe3-4', 'Apoe4-4']

    cells = []
    print groups

    lastlen = 0
    for i, lengrp in enumerate(groups):
        cell = {'levels': i+1}
        print lastlen, lastlen+lengrp
        cell['scans'] = scans[lastlen:lastlen+lengrp]
        lastlen = lengrp + lastlen
        print lastlen
        cells.append(cell)

    print cells
    print 'Analysis name:', analysis_name

    centering = [1] * len(names)
    #print 'Scans (%s):'%len(scans), scans
    print 'Vectors (%s)'%len(vectors)
    print 'Names (%s):'%len(names), names
    print 'Contrasts (%s):'%len(contrasts), contrasts

    covariates = []
    interactions = {'age':2, 'agesq':2}
    for name, v, c in zip(names, vectors, centering):
        covariates.append(dict(name=name, centering=c, interaction=interactions.get(name, 1), vector=v))

    factors = {'name': 'genotype', 'levels': 5, 'dept':0, 'variance': 1, 'gmsca':0, 'ancova':0}
    model = FullFactorialDesign(cells = cells, factors=factors,
                                    covariates = covariates,
                                    explicit_mask_file = explicitmask, contrasts=False)

    # Model Estimation
    est = spm.EstimateModel(estimation_method = {'Classical': 1})

    # Contrast Estimation
    con = spm.EstimateContrast(contrasts = contrasts,
                               group_contrast = True)

    # Creating Workflow
    a = pe.Workflow(name=analysis_name)
    a.base_dir = destdir

    n1 = pe.Node(model, name='modeldesign')
    n2 = pe.Node(est, name='estimatemodel')
    n3 = pe.Node(con, name='estimatecontrasts')

    a.connect([(n1, n2, [('spm_mat_file','spm_mat_file')] ),
               (n2,n3, [('spm_mat_file', 'spm_mat_file'),
                        ('beta_images', 'beta_images'),
                        ('residual_image', 'residual_image')]), ])
    a.config['execution']['stop_on_first_rerun'] = True
    return a


# == MAIN ==

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
	    description=textwrap.dedent('''\
                    Runs an SPM Multiple Regression full analysis using data from a given table as regressors,
                    using a given explicit mask and writes the results in a given directory.
                    Consists in three steps: Model design / Model estimation / Contrast estimation
	    '''))

    parser.add_argument("excel", type=str, help='Excel file containing the model data')
    parser.add_argument("destdir", type=str, help='Destination directory')
    parser.add_argument("--mask", type=str, help='Explicit mask used in the analysis', required=False, default='/home/grg/spm/MNI_T1_brain_mask.nii')
    parser.add_argument("--design", type=str, help='Design (0: Multiple Regression - 1: Full Factorial)', required=False, default=0)

    parser.add_argument("-v", dest='verbose', action='store_true', required=False, default=True)
    args = parser.parse_args()

    excel = args.excel
    mask = args.mask
    destdir = args.destdir
    design = args.design

    print 'Excel file:', excel
    print 'mask file:', mask
    print 'destination directory:', destdir

    if design==1:
        print 'Full Factorial Design'
        a = fullfactorial(excel, destdir, mask, analysis_name='test_fullfact')
        a.run()
    else:
        import os.path as osp
        aname = osp.splitext(osp.split(excel)[1])[0]
        a = generic_version(excel, destdir, mask, analysis_name=aname)
        a.run()
