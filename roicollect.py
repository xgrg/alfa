from matplotlib import pyplot as plt
import string, json
import pandas as pd
from glob import glob
import os.path as osp
import numpy as np

def getdict(data, key, column, column2=None, value=None):
    ''' From a Pandas DataFrame returns a dictionary given a key and a column,
    with the possibility of filtering on the value of a second column'''
    import string
    if not value is None:
        data = data[data[column2] == value]

    d1 = dict([(string.atoi(str(int(e))), v)
        for e, v in data[[key, column]].to_dict(orient='split')['data']])

    return d1

def collect_roivalues(roilabel, src=None, verbose=False):
    ''' This simply collects the values from *_stats.csv files and returns a
    DataFrame with these ROI values, given a specific `roilabel`.
    `src` is the folder containing the csv files.
    '''
    import string
    if src is None:
        src = '/tmp/roivalues_csf.5/' # new csv with the artifactual roi (temporal)
        print 'Warning: default csv files loaded from ', src

    subjects = [string.atoi(osp.split(e)[-1].split('_')[0])\
            for e in glob(osp.join(src, '*.csv'))]
    print len(subjects), 'subjects displayed'

    data = []
    subj = []
    for s in subjects:
        fp = osp.join(src, '%s_stats.csv'%s)
        if verbose:
            print s, fp
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
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    model = 'roi ~ 1 + C(apo) + gender + educyears + ventricles'
    test_scores = ols(model, data=df).fit()

    err = test_scores.predict(df) - df['roi']
    ycorr = np.mean(df['roi']) - err

    return ycorr

def set_figaxes(df):
    plt.xlabel('age')
    plt.ylabel('roi')
    plt.ylim([0.0007, 0.0010]) #df['roi'].max()])
    plt.xlim([df['age'].min(), df['age'].max()])


def get_groups(dataset, groups_names=[]):
    # take each group separately
    if len(groups_names) == 0:
        groups_names = ['apoe23', 'apoe24', 'apoe33', 'apoe34', 'apoe44']
    groups1 = []
    for i in xrange(5):
        groups1.append(dataset[dataset['apo'] == i])
    groups = []
    for name in groups_names:
        if name == 'C':
            groups.append(pd.concat([groups1[i] for i in [1,3,4]]))
        elif name == 'NC':
            groups.append(pd.concat([groups1[i] for i in [0,2]]))
        elif name == 'HO':
            groups.append(pd.concat([groups1[i] for i in [4]]))
        elif name == 'HT':
            groups.append(pd.concat([groups1[i] for i in [1,3]]))
        elif name == 'All':
            groups.append(pd.concat([groups1[i] for i in [0,1,2,3]]))
    if len(groups) == 0:
        return groups1, groups_names
    return groups, groups_names


def plot_region(roiname, dataset, order=1, ax=None):
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    if ax == None:
        fig = plt.figure(figsize=(6, 6))
        set_figaxes(dataset)
        ax = fig.add_subplot(111)

    colors = ['#800000','#003366','g','m','y']
    facecolors = ['#ff9999','#99ccff','g','m','y']
    formulas = ['roi ~ 1 + age',
                'roi ~ 1 + age + I(age**2)',
                'roi ~ 1 + age + I(age**2) + I(age**3)']
    print 'Region:', roiname, '- Fitting order:', order, '- Formula:', formulas[order-1]

    groups_names = ['HO', 'All']
    groups, groups_names = get_groups(dataset, groups_names = groups_names)


    for i, df in enumerate(groups):
        ax.scatter(df['age'], df['roi'], edgecolors=colors[i],
                   facecolors=facecolors[i], linewidth=0.5,
                   label='%s'%groups_names[i].capitalize(), s=20, alpha=0.7)
        x = pd.DataFrame({'age': np.linspace(df['age'].min(), df['age'].max(), 100)})

        poly = ols(formula=formulas[order-1], data=df).fit()
        ypred = poly.predict(x)
        #print np.std(ypred)
        ax.plot(x['age'], ypred, color=colors[i], linestyle='-',
                label='%s n=%s $R^2$=%.2f $AIC$=%.2f $\sigma$=%.3e'
                 % (groups_names[i].capitalize(), order, poly.rsquared, poly.aic, np.std(ypred)),
                alpha=1.0)

    ax.legend(prop={'size':8})
    ax.text(0.15, 0.95, roiname, horizontalalignment='center',
        verticalalignment='center', transform = ax.transAxes)
    plt.title(roiname)


def plot_regions(data, regions, labels=None, nb_orders=3):
    nb_regions = len(regions)
    fig = plt.figure(figsize=(8*nb_orders, 8*nb_regions), dpi=300, facecolor='white')

    if labels is None:
        print 'WARNING: using default value for labels'
        labels = ['left_occip', 'left_temporal', 'left_temporal2',
            'right_perihorn', 'left_occip2', 'left_perihorn', 'left_wm',
            'right_occip', 'right_temporal', 'left_insula']

    for i, roilabel in enumerate(regions):
        roiname = '%s'%labels[roilabel-1]
        # 1-st order
        ax = fig.add_subplot(nb_regions, nb_orders, nb_orders*i + 1)
        roivalues = collect_roivalues(roilabel)
        df = data.join(roivalues)
        print np.std(df['roi'])
        ycorr = pd.DataFrame(correct(df), columns=['roi'])
        df = data.join(ycorr)
        df['subject'] = df.index
        df = df.sort_values(['apo', 'subject']).dropna()

        set_figaxes(df)
        print np.std(df['roi'])
        plot_region(roiname, df, order=1, ax=ax)
