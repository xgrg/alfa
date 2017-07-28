from matplotlib import pyplot as plt
import string, json
import pandas as pd
from glob import glob
import os.path as osp
import numpy as np
import seaborn
from matplotlib.font_manager import FontProperties
import statsmodels.api as sm
from statsmodels.formula.api import ols

def getdict(data, key, column, column2=None, value=None):
    ''' From a Pandas DataFrame returns a dictionary given a key and a column,
    with the possibility of filtering on the value of a second column'''
    if not value is None:
        data = data[data[column2] == value]

    d1 = dict([(string.atoi(str(int(e))), v)
        for e, v in data[[key, column]].to_dict(orient='split')['data']])

    return d1

def collect_roivalues(roilabel, csvfiles, subjects, verbose=False):
    ''' This simply collects the values from *_stats.csv files and returns a
    DataFrame with these ROI values, given a specific `roilabel`.
    `subjects` is the list of subjects associated to these files (required in
    order to join information at a later step).
    '''
    data = []
    subj = []

    assert(len(csvfiles) == len(subjects))
    for s, fp in zip(subjects, csvfiles):
        #fp = osp.join(src, filepattern%s)
        #if verbose:
        #    print s, fp
        #if not osp.isfile(fp):
        #    if len(glob(fp)) != 1:
        #        raise AssertionError('multiple files found %s'%str(glob(fp)))
        #    fp = glob(fp)[0]  #in case a glob string is passed
        df = pd.read_csv(fp, sep='\t').set_index('ROI_label')
        try:
            data.append(df.ix[roilabel]['mean'])
            subj.append(s)
        except KeyError as e:
            print 'KeyError: skipped subject', s, 'label', e
        except IndexError as e:
            print 'IndexError: skipped subject', s, 'label', e
    return pd.DataFrame(data, index=subj, columns=['roi'])

def correct(df):
    ''' Applies a correction for covariates to a given DataFrame'''

    model = 'roi ~ 1 + gender + educyears + apo'
    print 'Model used for correction:', model
    test_scores = ols(model, data=df).fit()

    err = test_scores.predict(df) - df['roi']
    ycorr = np.mean(df['roi']) - err

    return ycorr

def set_figaxes(df, ylim=None):
    font = FontProperties(family='sans-serif', weight='heavy')
    plt.xlabel('age', fontproperties=font)
    plt.ylabel('roi', fontproperties=font)
    if not ylim is None:
        plt.ylim(ylim)
    #else:
     #df['roi'].max()])
    plt.xlim([df['age'].min(), df['age'].max()])


def get_groups(dataset, groups_names):
    '''Splits a dataset according to genotypic groups. Returns a list of
    DataFrames plus a list of group names.'''
    # take each group separately
    groups1 = []
    for i in xrange(5):
        groups1.append(dataset[dataset['apo'] == i])
    groups = []
    groups_ht = {'C': [1,3,4],
                 'NC': [0,2],
                 'HO': [4],
                 'HT': [1,3],
                 'notHO':[0,1,2,3],
                 'apoe44': [4],
                 'apoe34': [3],
                 'apoe33': [2],
                 'apoe24': [1],
                 'apoe23': [0],
                 'All':[0,1,2,3,4]}
    for name in groups_names:
        for k,v in groups_ht.items():
            if name == k:
                groups.append(pd.concat([groups1[i] for i in v]))
    if len(groups) == 0:
        return groups1
    return groups


def plot_region(dataset, roi_name='', groups=None, order=1, ax=None, ylim=[0.0005, 0.0010], c=None):
    from scipy import interpolate

    if ax == None:
        fig = plt.figure(figsize=(6, 6))
        set_figaxes(dataset, ylim=[dataset['roi'].min(), dataset['roi'].max()])
        ax = fig.add_subplot(111)

    font = FontProperties(family='sans-serif', weight='light')

    #TODO: dictionaries for always matched pairs groups/colors
    edgecolors = ['#800000','#003366','#ff8000','#cc6699','#33cc33']
    facecolors = ['#ff9999','#99ccff','#ffd699','#ecc6d9','#adebad']

    formulas = ['roi ~ 1 + age',
                'roi ~ 1 + age + I(age**2)',
                'roi ~ 1 + age + I(age**2) + I(age**3)']

    print 'Region:', roi_name#, '- Fitting order:', order, '- Formula:', formulas[order-1]

    # Splits the dataset into genotypic groups
    if groups is None:
        print 'WARNING: using the 5 genotypic groups'
        groups = ['apoe44', 'apoe34', 'apoe24', 'apoe33', 'apoe23']
    groups_sub = get_groups(dataset, groups_names = groups)

    for i, df in enumerate(groups_sub):
        # Plots the group cloud
        eg = edgecolors[i] if c==None else edgecolors[i+1]
        fg = facecolors[i] if c==None else facecolors[i+1]
        ax.scatter(df['age'], df['roi'], edgecolors=eg, facecolors=fg, linewidth=0.5,
                   label='%s'%groups[i].capitalize(), s=20, alpha=0.7)

        # Fits a line on the group data
        x = pd.DataFrame({'age': np.linspace(df['age'].min(), df['age'].max(), 500)})
        poly = ols(formula=formulas[order-1], data=df).fit()
        ypred = poly.predict(x)

        # Draws the fitted line
        ax.plot(x['age'], ypred, color=eg, linestyle='-',
                label='%s $\sigma$=%.3e' # n=%s $R^2$=%.2f $AIC$=%.2f '
                 % (groups[i].capitalize(), np.std(ypred)), #, order, poly.rsquared, poly.aic),
                alpha=1.0)

    # Legend, text, title...
    ax.legend(prop={'size':8})
    ax.text(0.15, 0.95, roi_name, horizontalalignment='center',
        verticalalignment='center', transform = ax.transAxes,
        fontproperties=font)
    plt.title(roi_name, fontproperties=font)


def plot_regions(data, labels, csvfiles, subjects, names=None, groups=None,
        nb_orders=1, do_correct=True, ylim=[0.0005, 0.0010]):
    ''' Generates a plot of ROI values to be looked up in a set of csvfiles,
    given a set of `labels`, a set of `subjects`.
    `names` is a dictionary naming the ROIs
    `groups` refers to the way the data should be grouped and displayed
    (according to genotypes)
    `nb_orders` (default: 1) sets how many orders/models to be fitted to the data
    `do_correct` (default: True) applies correction for covariates if provided
    in the table `data` (i.e. genotype, gender, education, ventricular volume)
    `ylim` sets y-limits on the final plot. '''

    fig = plt.figure(figsize=(8*nb_orders*2, 12*len(labels)), dpi=300, facecolor='white')

    # Managing options
    if names is None:
        print 'WARNING: using default values for labels'
        names = {1:'left_occip', 2:'left_temporal', 3:'left_temporal2',
            4:'right_perihorn', 5:'left_occip2', 6:'left_perihorn', 7:'left_wm',
            8:'right_occip', 9:'right_temporal', 10:'left_insula'}
        print names
    if groups is None:
        groups = ['HO', 'HT', 'NC']
        print 'using default groups'

    # Iterate on labels
    for i, roi_label in enumerate(labels):
        roi_name = '%s'%names[roi_label]

        # Iterate on orders
        for order in range(1, nb_orders+1):

            # 1-st order
            ax = fig.add_subplot(len(labels)*2, nb_orders*2, 2*nb_orders*i + order)

            # Fetch values for the given ROI and correct them for covariates
            roivalues = collect_roivalues(roi_label, csvfiles=csvfiles, subjects=subjects)
            df = data.join(roivalues).dropna()

            print 'Standard deviation of label %s:'%roi_name,\
                        np.std(df['roi'])
            if do_correct:
                ycorr = pd.DataFrame(correct(df), columns=['roi'])
                df = data.join(ycorr)
                df['subject'] = df.index
                df = df.sort_values(['apo', 'subject']).dropna()
                print 'Standard deviation after correction for covariates:',\
                        np.std(df['roi'])
            else:
                df['subject'] = df.index
                df = df.sort_values(['apo', 'subject']).dropna()

            # Plots the corrected values and fits a line over them
            set_figaxes(df, ylim=ylim) # Adjusts the axes according to the values
            plot_region(df, roi_name, order=order, groups=groups, ax=ax)
            ax = fig.add_subplot(len(labels)*2, nb_orders*2, 2*nb_orders*i + order + 1 )
            boxplot_region(df, groups, ax=ax)
    fig.savefig('/tmp/fig.svg')

def plot_significance(df, x1, x2, groups):
    # statistical annotation
    import scipy

    T = scipy.stats.ttest_ind(df[df['group']==groups[x1]]['roi'],
            df[df['group']==groups[x2]]['roi'])
    print T.pvalue
    y, h, col = df['roi'].mean() + df['roi'].std() + 1e-4, 1e-5, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    opt = {'ha':'center', 'va':'bottom', 'color':col}
    if T.pvalue<0.05:
        opt['weight'] = 'bold'
    plt.text((x1+x2)*.5, y+h, '%.3f'%T.pvalue, **opt)

def boxplot_region(df, groups, ax=None):
    import seaborn as sns
    # Plots the corrected values and fits a line over them
    groups_sub = get_groups(df, groups_names = groups)
    if ax == None:
        pass
        #fig = plt.figure(figsize=(6, 6))
        #set_figaxes(df, ylim=[df['roi'].min(), df['roi'].max()])
        #ax = fig.add_subplot(111)

    grp = []
    for i, each in enumerate(groups_sub):
        each['group'] = len(each['apo']) * [groups[i]]
        grp.append(each)
    df = pd.concat(grp)
    sns.boxplot(x='group', y='roi', data=df, showfliers=False)#, ax=ax)
    import itertools
    for i1, i2 in itertools.combinations(xrange(len(groups)), 2):
        plot_significance(df, i1, i2, groups)
