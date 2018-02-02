import pandas as pd
import numpy as np

def getdict(data, key, column, column2=None, value=None):
    import string
    if not value is None:
        data = data[data[column2] == value]

    d1 = dict([(string.atoi(str(int(e))), v) for e, v in data[[key, column]].to_dict(orient='split')['data']])
    return d1

def build_matrix(images, covlist, covtable, subjects=None):
    '''Returns a table with a column for paths to `images`, then a series of
    `covariates` taken from an Excel table `covtable`, restricted to a set of
    `subjects`.
    If `subjects` not provided, then a `subject` will be looked for in the
    Excel table.'''
    cov_sub = covtable['subject'].tolist()
    if subjects is None:
        print 'Will take all subjects available'
        subjects = cov_sub
    diff_sub = set(subjects).difference(set(cov_sub))
    if len(diff_sub) != 0:
        print diff_sub, 'mismatching'
        assert(len(diff_sub) == 0)

    assert(len(images)==len(subjects))

    data = []
    for im, s in zip(images, subjects):
        if not str(s) in im:
            print s, 'not in', im, 'you should check your data.'
        row = [im]
        for e in covlist:
            row.append(covtable[covtable['subject']==s][e].values[0])
        data.append(row)

    col = ['images']
    col.extend(covlist)
    df = pd.DataFrame(data, columns=col)
    return df

def build_tbss_matrix(df):
    ''' Returns a TBSS-ready design matrix with corresponding contrasts'''

    if 'images' in df.columns:
        del df['images']
    covlist = df.columns
    mat = ['/NumWaves %s'%len(covlist)]
    mat.append('/NumPoints %s'%len(df))
    mat.append('/Matrix')
    for row_index, row in df.iterrows():
        s1 = ' '.join([str(row[e]) for e in covlist])
        mat.append(s1)

    con = ['/NumWaves %s'%len(covlist)]

    for i, e in enumerate(covlist):
        con.append('/ContrastName%s %s (+)'%(str(2*i), e))
        con.append('/ContrastName%s %s (-)'%(str(2*i+1), e))
    con.append('/NumContrasts %s'%(str(2*len(covlist))))
    con.append('/Matrix')
    for i, e in enumerate(covlist):
        c = [0] * len(covlist)
        c[i] = 1
        con.append(' '.join([str(each) for each in c]))
        c[i] = -1
        con.append(' '.join([str(each) for each in c]))

    return '\n'.join(mat), '\n'.join(con)

def build_interaction(df, var, categ_var):
    '''Adds columns to a DataFrame with the interaction between a variable and
    a categorical variable.'''

    groups = np.unique(df['%s'%categ_var].values).tolist()
    # particular case for apoe
    grp_labels = {0:'23', 1:'24', 2:'33', 3:'34', 4:'44'} \
        if categ_var == 'apo' else groups

    for i in groups:
        apo = pd.DataFrame(df[categ_var] == i, dtype=np.int)
        apocol = '%s%s'%(categ_var, grp_labels[i])
        df[apocol] = apo
        intercol = '%s%s'%(var, grp_labels[i])
        df[intercol] = df.apply(lambda row: row[apocol]*row[var], axis=1)
        del df[apocol]
    return df

def build_dummy(df, categ_var):
    '''Adds columns to a DataFrame with dummy variables from a categorical
    variable.'''

    groups = np.unique(df['%s'%categ_var].values).tolist()
    # particular case for apoe
    grp_labels = {0:'Apoe2-3', 1:'Apoe2-4', 2:'Apoe3-3', 3:'Apoe3-4', 4:'Apoe4-4'} \
        if categ_var == 'apo' else groups

    for i in groups:
        apo = pd.DataFrame(df[categ_var] == i, dtype=np.int)
        apocol = grp_labels[i]
        df[apocol] = apo
    return df
