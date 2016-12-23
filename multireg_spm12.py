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

# === Building regressors with data from a provided Excel file ===

def make_vectors_original_version(data, verbose=True):

    names = ['Apoe2-3', 'Apoe2-4', 'Apoe3-3', 'Apoe3-4', 'Apoe4-4',
                    'age23', 'age24', 'age33', 'age34', 'age44',
                    'agesq23', 'agesq24', 'agesq33', 'agesq34', 'agesq44',
                    'Gender(0=female)', 'Years of Education']
    if verbose:
        print 'Columns used in the model:', names

    # Model Design
    vectors = [data[each].tolist() for each in names]

    return vectors, names

def make_vectors_original_version_with_ventricles_FS(data, verbose=True):

    names = ['Apoe2-3', 'Apoe2-4', 'Apoe3-3', 'Apoe3-4', 'Apoe4-4',
                    'age23', 'age24', 'age33', 'age34', 'age44',
                    'agesq23', 'agesq24', 'agesq33', 'agesq34', 'agesq44', 'ventricles_FS',
                    'Gender(0=female)', 'Years of Education']
    if verbose:
        print 'Columns used in the model:', names

    # Model Design
    vectors = [data[each].tolist() for each in names]

    return vectors, names

def make_vectors_original_version_with_ventricles_JDG(data, verbose=True):

    names = ['Apoe2-3', 'Apoe2-4', 'Apoe3-3', 'Apoe3-4', 'Apoe4-4',
                    'age23', 'age24', 'age33', 'age34', 'age44',
                    'agesq23', 'agesq24', 'agesq33', 'agesq34', 'agesq44', 'ventricles_JDG',
                    'Gender(0=female)', 'Years of Education']
    if verbose:
        print 'Columns used in the model:', names

    # Model Design
    vectors = [data[each].tolist() for each in names]

    return vectors, names

def make_vectors_just_age(data, verbose=True):
    ''' This version removes the genotype from the model and performs an F-test for
    the main effect of age on Mean diffusivity'''

    names = [ 'age23', 'age24', 'age33', 'age34', 'age44',
                    'agesq23', 'agesq24', 'agesq33', 'agesq34', 'agesq44', 'ventricles_FS',
                    'Gender(0=female)', 'Years of Education']

    # Model Design
    vectors = [data[each].tolist() for each in names]
    agesq = [sum([vectors[names.index(e)][i] for e in ['agesq23', 'agesq24', 'agesq33', 'agesq34', 'agesq44']]) for i in xrange(len(vectors[names.index('agesq23')]))]
    age = [sum([vectors[names.index(e)][i] for e in ['age23', 'age24', 'age33', 'age34', 'age44']]) for i in xrange(len(vectors[names.index('age23')]))]
    vectors = [age, agesq, vectors[names.index('Gender(0=female)')], vectors[names.index('Years of Education')]]
    names = ['age', 'agesq', 'Gender(0=female)', 'Years of Education']

    if verbose:
        print 'Columns used in the model:', names

    return vectors, names

# === Building contrasts vectors depending on the analysis ===

def make_contrasts_original_version():
    cont1 = ('Apo2-3>Apo2-4', 'T', ['Apoe2-3', 'Apoe2-4'], [1,-1])
    cont2 = ('Apo2-4>Apo3-3', 'T', ['Apoe2-4', 'Apoe3-3'], [1,-1])
    cont3 = ('Apo3-3>Apo3-4', 'T', ['Apoe3-3', 'Apoe3-4'], [1,-1])
    cont4 = ('Apo3-4>Apo4-4', 'T', ['Apoe3-4', 'Apoe4-4'], [1,-1])
    cont5 = ('Main effect ApoE', 'F', [cont1, cont2, cont3, cont4])
    cont6 = ('C<NC', 'T', ['Apoe2-3', 'Apoe2-4', 'Apoe3-3', 'Apoe3-4', 'Apoe4-4'], [3, -2, 3, -2, -2])
    cont7 = ('C>NC', 'T', ['Apoe2-3', 'Apoe2-4', 'Apoe3-3', 'Apoe3-4', 'Apoe4-4'], [-3, 2, -3, 2, 2])
    cont8 = ('HO<HZ', 'T', ['Apoe2-3', 'Apoe2-4', 'Apoe3-3', 'Apoe3-4', 'Apoe4-4'], [1, 1, 1, 1, -4])
    cont9 = ('HO>HZ', 'T', ['Apoe2-3', 'Apoe2-4', 'Apoe3-3', 'Apoe3-4', 'Apoe4-4'], [-1, -1, -1, -1, 4])
    cont10 = ('HO<All_Age_Squared', 'T', ['agesq23', 'agesq24', 'agesq33', 'agesq34', 'agesq44'], [3, -2, 3, -2, -2])
    cont11 = ('HO>All_Age_Squared', 'T', ['agesq23', 'agesq24', 'agesq33', 'agesq34', 'agesq44'], [-3, 2, -3, 2, 2])
    contrasts = [cont1, cont2, cont3, cont4, cont5, cont6, cont7, cont8, cont9, cont10, cont11]
    return contrasts

def make_contrasts_just_age():
    contT = ('Main effect Agesq', 'T', names, [0,1,0,0])
    contF = ('Main effect Agesq', 'F', [contT])
    contrasts = [contT, contF]
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

    $param$ identifies a column from the Excel

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

def original_version(param, excel_file, destdir, explicitmask, analysis_name):

    print 'Analysis name:', analysis_name

    data = pd.read_excel(excel_file)
    scans = data[param].tolist()

    vectors, names = make_vectors_original_version(data)
    contrasts = make_contrasts_original_version()

    a = multiple_regression_analysis(scans, vectors, names, contrasts, destdir, explicitmask, analysis_name)
    return a

def original_version_ventFS(param, excel_file, destdir, explicitmask, analysis_name):

    print 'Analysis name:', analysis_name

    data = pd.read_excel(excel_file)
    scans = data[param].tolist()

    vectors, names = make_vectors_original_version_with_ventricles_FS(data)
    contrasts = make_contrasts_original_version()

    a = multiple_regression_analysis(scans, vectors, names, contrasts, destdir, explicitmask, analysis_name)
    return a

def original_version_ventJDG(param, excel_file, destdir, explicitmask, analysis_name):

    print 'Analysis name:', analysis_name

    data = pd.read_excel(excel_file)
    scans = data[param].tolist()

    vectors, names = make_vectors_original_version_with_ventricles_JDG(data)
    contrasts = make_contrasts_original_version()

    a = multiple_regression_analysis(scans, vectors, names, contrasts, destdir, explicitmask, analysis_name)
    return a

def fullfactorial(param, excel_file, destdir, explicitmask, analysis_name):

    print 'Analysis name:', analysis_name
    import json

    data = pd.read_excel(excel_file)
    scans = data[param].tolist()

    vectors, names = make_vectors_just_age(data)

    contrasts = make_contrasts_fullfactorial_version()

    import numpy as np
    groups = [np.sum(data['Apoe2-3']),
              np.sum(data['Apoe2-4']),
              np.sum(data['Apoe3-3']),
              np.sum(data['Apoe3-4']),
              np.sum(data['Apoe4-4'])]


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

    parser.add_argument("param", type=str, help='Type of parametric maps to run the analysis on')
    parser.add_argument("excel", type=str, help='Excel file containing the model data')
    parser.add_argument("destdir", type=str, help='Destination directory')
    parser.add_argument("--mask", type=str, help='Explicit mask used in the analysis', required=False, default='/home/grg/spm/MNI_T1_brain_mask.nii')

    parser.add_argument("-v", dest='verbose', action='store_true', required=False, default=True)
    args = parser.parse_args()

    param = args.param
    excel = args.excel
    mask = args.mask
    destdir = args.destdir

    print 'Excel file:', excel
    print 'mask file:', mask
    print 'destination directory:', destdir

#    print 'First analysis without ventricular volumes in the model'
#    for param in ['MD', 'MD_pred', 'MD_corr', 'Jacobians']: #, 'FA', 'L1', 'RD']:
#
#        a = original_version(param, excel, destdir, mask, analysis_name='%s_wo_ventvol'%param)
#        a.run()
#
#    print 'Second analysis with FreeSurfer ventricular volumes in the model'
#    for param in ['MD', 'MD_pred', 'MD_corr', 'Jacobians']: #, 'FA', 'L1', 'RD']:
#
#        a = original_version_ventFS(param, excel, destdir, mask, analysis_name='%s_w_FS_ventvol'%param)
#        a.run()
#
#    print 'Third analysis with JDG ventricular volumes in the model'
#    for param in ['MD', 'MD_pred', 'MD_corr', 'Jacobians']: #, 'FA', 'L1', 'RD']:
#
#        a = original_version_ventJDG(param, excel, destdir, mask, analysis_name='%s_w_JDG_ventvol'%param)
#        a.run()

    a = fullfactorial(param, excel, destdir, mask, analysis_name='test_fullfact')
    a.run()
