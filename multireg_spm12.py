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
        cont8 = ('HO<All', 'T', ['Apoe2-3', 'Apoe2-4', 'Apoe3-3', 'Apoe3-4', 'Apoe4-4'], [1, 1, 1, 1, -4])
        cont9 = ('HO>All', 'T', ['Apoe2-3', 'Apoe2-4', 'Apoe3-3', 'Apoe3-4', 'Apoe4-4'], [-1, -1, -1, -1, 4])
        cont10 = ('HO<HT', 'T', ['Apoe2-4', 'Apoe3-4', 'Apoe4-4'], [1, 1, -2])
        cont11 = ('HO>HT', 'T', ['Apoe2-4', 'Apoe3-4', 'Apoe4-4'], [-1, -1, 2])
        cont12 = ('HO<NC', 'T', ['Apoe2-3', 'Apoe3-3', 'Apoe4-4'], [1, 1, -2])
        cont13 = ('HO>NC', 'T', ['Apoe2-3', 'Apoe3-3', 'Apoe4-4'], [-1, -1, 2])
        cont14 = ('HT<NC', 'T', ['Apoe2-3', 'Apoe2-4', 'Apoe3-3', 'Apoe3-4'], [1, -1, 1, -1])
        cont15 = ('HT>NC', 'T', ['Apoe2-3', 'Apoe2-4', 'Apoe3-3', 'Apoe3-4'], [-1, 1, -1, 1])
        cont16 = ('Dose-dependent effect', 'T', ['Apoe3-3', 'Apoe3-4', 'Apoe4-4'], [-1, 0, 1])

        contrasts.extend([cont1, cont2, cont3, cont4, cont5, cont6, cont7, cont8, cont9, cont10, cont11, cont12, cont13, cont14, cont15, cont16])

    if 'age23' in names:
        print 'Interaction linear age-genotype detected'

        cont17 = ('age23>age24', 'T', ['age23', 'age24'], [1,-1])
        cont18 = ('age24>age33', 'T', ['age24', 'age33'], [1,-1])
        cont19 = ('age33>age34', 'T', ['age33', 'age34'], [1,-1])
        cont20 = ('age34>age44', 'T', ['age34', 'age44'], [1,-1])
        cont21 = ('Interaction linear age-genotype', 'F', [cont17, cont18, cont19, cont20])
        cont22 = ('C<NC_Age_Linear', 'T', ['age23', 'age24', 'age33', 'age34', 'age44'], [3, -2, 3, -2, -2])
        cont23 = ('C>NC_Age_Linear', 'T', ['age23', 'age24', 'age33', 'age34', 'age44'], [-3, 2, -3, 2, 2])
        cont24 = ('HO<All_Age_Linear', 'T', ['age23', 'age24', 'age33', 'age34', 'age44'], [1, 1, 1, 1, -4])
        cont25 = ('HO>All_Age_Linear', 'T', ['age23', 'age24', 'age33', 'age34', 'age44'], [-1, -1, -1, -1, 4])
        cont26 = ('HO<HT_Age_Linear', 'T', ['age24', 'age34', 'age44'], [1, 1, -2])
        cont27 = ('HO>HT_Age_Linear', 'T', ['age24', 'age34', 'age44'], [-1, -1, 2])
        cont28 = ('HO<NC_Age_Linear', 'T', ['age23', 'age33', 'age44'], [1, 1, -2])
        cont29 = ('HO>NC_Age_Linear', 'T', ['age23', 'age33', 'age44'], [-1, -1, 2])
        cont30 = ('HT<NC_Age_Linear', 'T', ['age23', 'age24', 'age33', 'age34'], [1, -1, 1, -1])
        cont31 = ('HT>NC_Age_Linear', 'T', ['age23', 'age24', 'age33', 'age34'], [-1, 1, -1, 1])

        contrasts.extend([cont17, cont18, cont19, cont20, cont21, cont22, cont23, cont24, cont25, cont26, cont27, cont28, cont29, cont30, cont31])

    if 'agesq23' in names:
        print 'Interaction quadratic age-genotype detected'

        cont32 = ('agesq23>agesq24', 'T', ['agesq23', 'agesq24'], [1,-1])
        cont33 = ('agesq24>agesq33', 'T', ['agesq24', 'agesq33'], [1,-1])
        cont34 = ('agesq33>agesq34', 'T', ['agesq33', 'agesq34'], [1,-1])
        cont35 = ('agesq34>agesq44', 'T', ['agesq34', 'agesq44'], [1,-1])
        cont36 = ('Interaction Age square-genotype', 'F', [cont32, cont33, cont34, cont35])
        cont37 = ('C<NC_Age_Square', 'T', ['agesq23', 'agesq24', 'agesq33', 'agesq34', 'agesq44'], [3, -2, 3, -2, -2])
        cont38 = ('C>NC_Age_Square', 'T', ['agesq23', 'agesq24', 'agesq33', 'agesq34', 'agesq44'], [-3, 2, -3, 2, 2])
        cont39 = ('HO<All_Age_Square', 'T', ['agesq23', 'agesq24', 'agesq33', 'agesq34', 'agesq44'], [1, 1, 1, 1, -4])
        cont40 = ('HO>All_Age_Square', 'T', ['agesq23', 'agesq24', 'agesq33', 'agesq34', 'agesq44'], [-1, -1, -1, -1, 4])

        contrasts.extend([cont32, cont33, cont34, cont35, cont36, cont37, cont38, cont39, cont40])

    if 'NC' in names and 'HT' in names and 'HO' in names:
        print 'Non-carriers / Heterozygotes / Homozygotes detected'

        cont41 = ('HO>HT', 'T', ['HO', 'HT'], [1,-1])
        cont42 = ('HO>NC', 'T', ['HO', 'NC'], [1,-1])
        cont43 = ('HT>NC', 'T', ['HT', 'NC'], [1,-1])

        contrasts.extend([cont41, cont42, cont43])

    if 'ageNC' in names and 'ageHT' in names and 'ageHO' in names:
        print 'Interaction with age Non-carriers / Heterozygotes / Homozygotes detected'

        cont44 = ('HO>HT_Age_Linear', 'T', ['ageHO', 'ageHT'], [1,-1])
        cont45 = ('HO>NC_Age_Linear', 'T', ['ageHO', 'ageNC'], [1,-1])
        cont46 = ('HT>NC_Age_Linear', 'T', ['ageHT', 'ageNC'], [1,-1])

        contrasts.extend([cont44, cont45, cont46])


    cov = {'age': 'Linear age',
            'agesq': 'Age square',
            'educyears': 'Educational Years',
            'gender': 'Gender'}

    for k, v in cov.items():
        if k in names:
            print 'Effect of %s'%v
            c = ('Effect %s'%v, 'T', [k], [1])
            contrasts.append(c)

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
                                    explicit_mask_file = explicitmask,
                                    use_implicit_threshold = True
                                    )

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
    a.config['execution']['remove_unnecessary_outputs'] = False
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
    parser.add_argument("--mask", type=str, help='Explicit mask used in the analysis',
        required=False,
        default='/home/grg/spm/MNI_T1_brain_wo_csf.nii')
    parser.add_argument("--design", type=str,
        help='Design (0: Multiple Regression - 1: Full Factorial)',
        required=False, default=0)
    parser.add_argument("-v", dest='verbose', action='store_true',
        required=False, default=True)
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
