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
        df = pd.read_csv(fp, sep='\t').set_index('ROI_label')
        try:
            data.append(df.ix[roilabel]['mean'])
            subj.append(s)
        except KeyError as e:
            print 'KeyError: skipped subject', s, 'label', e
        except IndexError as e:
            print 'IndexError: skipped subject', s, 'label', e
    return pd.DataFrame(data, index=subj, columns=['roi'])

def correct(df, model='roi ~ 1 + gender + age'): #educyears + apo' ):
    ''' Applies a correction for covariates to a given DataFrame'''
    print 'Model used for correction:', model
    test_scores = ols(model, data=df).fit()

    err = test_scores.predict(df) - df['roi']
    ycorr = np.mean(df['roi']) - err

    return ycorr

def set_figaxes(df, ylim=None, xlim=None):
    font = FontProperties(family='sans-serif', weight='heavy')
    plt.xlabel('age', fontproperties=font)
    plt.ylabel('roi', fontproperties=font)
    if not ylim is None:
        plt.ylim(ylim)
    plt.xlim([df['age'].min()-1, df['age'].max()+1])


def get_groups(dataset, groups_names, by='apo'):
    '''Splits a dataset according to genotypic groups. Returns a list of
    DataFrames plus a list of group names.'''
    groups1 = [dataset[dataset[by] == i] for i in xrange(5)]
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
                 'All':[0,1,2,3,4],
                 'm':[0],
                 'f':[1]}
    for name in groups_names:
        for k,v in groups_ht.items():
            if name == k:
                groups.append(pd.concat([groups1[i] for i in v]))
    if len(groups) == 0:
        return groups1
    return groups


def lm_plot(data, roivalues, ylim):
    df = data.join(roivalues).dropna()

    # correction without ajusting for age
    adj_model = 'roi ~ 1 + gender'
    ycorr = pd.DataFrame(correct(df, adj_model), columns=['roi'])
    df = data.join(ycorr)
    print 'Standard deviation after correction for model %s:'%adj_model,\
            np.std(df['roi'])
    df['subject'] = df.index
    df = df.sort_values(['apo', 'subject']).dropna()

    groups_sub = get_groups(df, groups_names = ['HO','HT','NC'], by='apo')
    gd = []
    for i, g in enumerate(groups_sub):
        g['apo'] = ['HO','HT','NC'][i]
        gd.append(g)
    df = pd.concat(gd)

    import seaborn as sns

    lm = sns.lmplot(x='age', y='roi',  data=df, size=6.2, aspect=1.35, ci=90, hue='apo', legend=False,
        palette={'HO':'#800000', 'HT':'#ff8000', 'NC':'#003366', 1: '#ffd699', 0:'#99ccff', 'notHO':'#99ccff', 'f':'#ff9999', 'm':'#99ccff',
        'apoe44':'#ff9999', 'apoe34':'#ffd699', 'apoe33':'#99ccff'}, truncate=True, sharex=False,sharey=False)
        #scatter_kws={'linewidths':1,'edgecolor':'#800000','#ff8000','#003366']})
    ax = lm.axes
    ax[0,0].set_ylim(ylim)
    ax[0,0].set_xlim([44,76])
    ax[0,0].set_yticklabels(['%.2e'%x for x in ax[0,0].get_yticks()])
    ax[0,0].tick_params(labelsize=12)
    ax[0,0].set_ylabel('')
    ax[0,0].set_xlabel('age', fontsize=15, weight='bold')
    #ax[0,0].set_yticklabels([]) #['%.2e'%x for x in ax[0,0].get_yticks()])
    #ax[0,0].set_xticklabels([])

    pal = {'HO':'#ff9999', 'HT':'#ffd699', 'NC':'#99ccff'}
    pal2 = {'HO':'#800000','HT':'#ff8000','NC':'#003366'}

    for i,row in enumerate(df.values):
        lm.ax.plot(row[2], row[-1], ms=5.5, mew=0.4, marker='o', ls='-', mfc=pal[row[1]], mec=pal2[row[1]], lw=1)
    lm.savefig('/tmp/fig.svg')

def plot_region(dataset, roi_name='', groups=None, by='apo', order=1, ax=None, ylim=[0.0005, 0.0010], c=None):
    from scipy import interpolate

    if ax == None:
        fig = plt.figure(figsize=(6, 6))
        set_figaxes(dataset, ylim=[dataset['roi'].min(), dataset['roi'].max()])
        ax = fig.add_subplot(111)

    font = FontProperties(family='sans-serif', weight='light')

    #TODO: dictionaries for always matched pairs groups/colors
    edgecolors = ['#800000','#ff8000','#003366','#cc6699','#33cc33']
    facecolors = ['#ff9999','#ffd699','#99ccff','#ecc6d9','#adebad']

    formulas = ['roi ~ 1 + age',
                'roi ~ 1 + age + I(age**2)',
                'roi ~ 1 + age + I(age**2) + I(age**3)']

    print 'Region:', roi_name#, '- Fitting order:', order, '- Formula:', formulas[order-1]

    # Splits the dataset into genotypic groups
    if groups is None:
        print 'WARNING: using the 5 genotypic groups'
        groups = ['apoe44', 'apoe34', 'apoe24', 'apoe33', 'apoe23']
    groups_sub = get_groups(dataset, groups_names = groups, by=by)

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
                label='%s'# $\sigma$=%.3e' # n=%s $R^2$=%.2f $AIC$=%.2f '
                 % (groups[i]),#, np.std(ypred)), #, order, poly.rsquared, poly.aic),
                alpha=1.0)
        ax.set_yticklabels(['%.2e'%x for x in ax.get_yticks()])


    # Legend, text, title...
    ax.legend(prop={'size':8})
    #ax.text(0.15, 0.95, roi_name, horizontalalignment='center',
    #    verticalalignment='center', transform = ax.transAxes,
    #    fontproperties=font)
    plt.title(roi_name, fontproperties=font)


def plot_regions(data, labels, csvfiles, subjects, names, groups, by,
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

    print names
    print 'groups', groups

    table = []
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

            # correction without ajusting for age
            adj_model = 'roi ~ 1 + gender'
            ycorr = pd.DataFrame(correct(df, adj_model), columns=['roi'])
            df = data.join(ycorr)
            print 'Standard deviation after correction for model %s:'%adj_model,\
                    np.std(df['roi'])
            df['subject'] = df.index
            df = df.sort_values(['apo', 'subject']).dropna()

            # Plots the corrected values and fits a line over them
            set_figaxes(df, ylim=ylim) # Adjusts the axes according to the values
            plot_region(df, roi_name, order=order, groups=groups, by=by, ax=ax)
            ax = fig.add_subplot(len(labels)*2, nb_orders*2, 2*nb_orders*i + order + 1 )


            # correction also ajusting for age
            adj_model = 'roi ~ 1 + gender + age'
            ycorr = pd.DataFrame(correct(df, adj_model), columns=['roi'])
            df = data.join(ycorr)
            print 'Standard deviation after correction for model %s:'%adj_model,\
                    np.std(df['roi'])
            df['subject'] = df.index
            df = df.sort_values(['apo', 'subject']).dropna()

            pval, hdr = boxplot_region(df, groups, by=by, ax=ax)
            row = [roi_name]
            row.extend(pval)
            table.append(row)
    fig.savefig('/tmp/fig.svg')
    columns = ['roi_name']
    columns.extend(hdr)
    return pd.DataFrame(table, columns=columns)

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
    return T.pvalue

def boxplot_region(df, groups, by='apo', ax=None):
    import seaborn as sns
    # Plots the corrected values and fits a line over them
    groups_sub = get_groups(df, groups_names = groups, by=by)
    if ax == None:
        pass

    grp = []
    for i, each in enumerate(groups_sub):
        each['group'] = len(each[by]) * [groups[i]]
        grp.append(each)
    df = pd.concat(grp)

    pvals = []
    hdr = []
    box = sns.boxplot(x='group', y='roi', data=df, showfliers=False,
           palette={'HO':'#ff9999', 'HT':'#ffd699','NC':'#99ccff', 'notHO':'#99ccff', 'f':'#ff9999', 'm':'#99ccff',
           'apoe44':'#ff9999', 'apoe34':'#ffd699', 'apoe33':'#99ccff'})#, ax=ax)
    #box = sns.violinplot(x='group', y='roi', data=df, #, showfliers=False,
    #       palette={'HO':'#ff9999', 'HT':'#ffd699','NC':'#99ccff', 'notHO':'#99ccff', 'f':'#ff9999', 'm':'#99ccff',
    #       'apoe44':'#ff9999', 'apoe34':'#ffd699', 'apoe33':'#99ccff'})#, ax=ax)
    box.axes.set_yticklabels(['%.2e'%x for x in box.axes.get_yticks()])

    import itertools
    for i1, i2 in itertools.combinations(xrange(len(groups)), 2):
        pval = plot_significance(df, i1, i2, groups)
        pvals.append(pval)
        hdr.append((groups[i1], groups[i2]))
    return pvals, hdr
