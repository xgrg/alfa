import seaborn as sns
import scipy as sc
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import logging as log

#import matplotlib as mpl
#mpl.rcParams['figure.facecolor'] = 'white'

def correct(df, model='roi ~ 1 + gender + age'): #educyears + apo' ):
    ''' Adjusts a variable according to a given model'''
    model = ols(model, data=df)
    depvar = model.endog_names
    test_scores = model.fit()
    err = test_scores.predict(df) - df[depvar]
    ycorr = np.mean(df[depvar]) - err

    return ycorr

def set_figaxes(df, x='age', y='roi', ylim=None, xlim=None):
    font = FontProperties(family='sans-serif', weight='heavy')
    plt.xlabel(x, fontproperties=font)
    plt.ylabel(y, fontproperties=font)
    if not ylim is None:
        plt.ylim(ylim)
    plt.xlim([df[x].min()-1, df[x].max()+1])


def plot_region(data, roi_name='', by='apoe', ax=None, ylim=[0.0005, 0.0010]):
    from scipy import interpolate

    if ax == None:
        fig = plt.figure(figsize=(6, 6))
        set_figaxes(data, ylim=[data[roi_name].min(), data[roi_name].max()])
        ax = fig.add_subplot(111)

    font = FontProperties(family='sans-serif', weight='light')

    #TODO: dictionaries for always matched pairs groups/colors
    edgecolors = ['#800000','#ff8000','#003366','#cc6699','#33cc33']
    facecolors = ['#ff9999','#ffd699','#99ccff','#ecc6d9','#adebad']
    formulas = ['roi ~ 1 + age']

    groups = data.groupby(by)

    for i in groups.groups.keys():
        df = pd.DataFrame(groups.get_group(i)).rename(columns={roi_name:'roi'})

        # Plots the group cloud
        eg = edgecolors[i]
        fg = facecolors[i]
        ax.scatter(df['age'], df['roi'], edgecolors=eg, facecolors=fg, linewidth=0.5,
                   label='%s=%s'%(by, i), s=20, alpha=0.7)

        # Fits a line on the group data
        x = pd.DataFrame({'age': np.linspace(df['age'].min(), df['age'].max(), 500)})
        poly = ols(formula=formulas[0], data=df).fit()
        ypred = poly.predict(x)

        # Draws the fitted line
        ax.plot(x['age'], ypred, color=eg, linestyle='-',
                label='%s=%s'% (by, i), alpha=1.0)
        ax.set_yticklabels(['%.2e'%x for x in ax.get_yticks()])


    # Legend, text, title...
    ax.legend(prop={'size':8})
    plt.title(roi_name, fontproperties=font)

def boxplot_region(df, groups, by='apo', ax=None):
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


def plot_significance(df, x1, x2, groups):
    # statistical annotation

    T = sc.stats.ttest_ind(df[df['group']==groups[x1]]['roi'],
            df[df['group']==groups[x2]]['roi'])
    print T.pvalue
    y, h, col = df['roi'].mean() + df['roi'].std() + 1e-4, 1e-5, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    opt = {'ha':'center', 'va':'bottom', 'color':col}
    if T.pvalue<0.05:
        opt['weight'] = 'bold'
    plt.text((x1+x2)*.5, y+h, '%.3f'%T.pvalue, **opt)
    return T.pvalue

def lm_plot(y, x, data, covariates=['gender', 'age'], hue='apoe', ylim=None,
        savefig=None, facecolor='white'):

    # building a new table with only needed variables
    # y is renamed to roi to avoid potential issues with strange characters
    roi_name = y
    log.info('Dependent variable: %s'%y)
    variables = {x, y}
    if not hue is None:
        variables.add(hue)
    for c in covariates:
        variables.add(c)
    df = pd.DataFrame(data, columns=list(variables)).rename(columns={y:'roi'})
    df = df.dropna()
    y = 'roi'

    if len(covariates) != 0:
        adj_model = 'roi ~ %s + 1'%'+'.join(covariates)
        log.info('Fit model used for correction: %s'%adj_model)

        # correcting depending variable
        ycorr = pd.DataFrame(correct(df, adj_model), columns=[y])
        del df[y]
        df = df.join(ycorr)

    # plotting
    lm = sns.lmplot(x=x, y=y,  data=df, size=6.2, hue=hue, aspect=1.35, ci=90,
	 truncate=True, sharex=False,sharey=False)
    ax = lm.axes
    if ylim is None:
        ax[0,0].set_ylim([df[y].min(), df[y].max()])
    else:
        ax[0,0].set_ylim(ylim)
    ax[0,0].set_xlim([df[x].min(), df[x].max()])
    ax[0,0].set_yticklabels(['%.2e'%i for i in ax[0,0].get_yticks()])
    ax[0,0].tick_params(labelsize=12)
    ax[0,0].set_ylabel('')
    ax[0,0].set_xlabel(x, fontsize=15, weight='bold')
    lm.fig.suptitle(roi_name)

    if not savefig is None:
        lm.savefig(savefig, facecolor=facecolor)
    return df
