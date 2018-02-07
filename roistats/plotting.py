import seaborn
import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = 'white'

def correct(df, model='roi ~ 1 + gender + age', verbose=False): #educyears + apo' ):
    ''' Applies a correction for covariates to a given DataFrame'''
    if verbose:
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

def lm_plot(data, roi_name, ylim=None):
    data = pd.DataFrame(data, columns=['apoe','age','gender', roi_name]).rename(columns={roi_name:'roi'})
    data = data.dropna()

    # correction without ajusting for age
    adj_model = 'roi ~ 1 + gender'
    ycorr = pd.DataFrame(correct(data, adj_model), columns=['roi'])
    del data['roi']
    df = data.join(ycorr)
    #print 'Standard deviation after correction for model %s:'%adj_model,\
    #        np.std(df['roi'])
    df['subject'] = df.index
    df = df.sort_values(['apoe', 'subject']).dropna()

    import seaborn as sns

    lm = sns.lmplot(x='age', y='roi',  data=df, size=6.2, aspect=1.35, ci=90, hue='apoe', legend=False,
        palette={2:'#800000', 1:'#ff8000', 0:'#003366'}, truncate=True, sharex=False,sharey=False)
        #scatter_kws={'linewidths':1,'edgecolor':'#800000','#ff8000','#003366']})
    ax = lm.axes
    if ylim is None:
        ax[0,0].set_ylim([df['roi'].min(), df['roi'].max()])
    ax[0,0].set_xlim([50,70])
    ax[0,0].set_yticklabels(['%.2e'%x for x in ax[0,0].get_yticks()])
    ax[0,0].tick_params(labelsize=12)
    ax[0,0].set_ylabel('')
    ax[0,0].set_xlabel('age', fontsize=15, weight='bold')
    #ax[0,0].set_yticklabels([]) #['%.2e'%x for x in ax[0,0].get_yticks()])
    #ax[0,0].set_xticklabels([])

    pal = {2:'#ff9999', 1:'#ffd699', 0:'#99ccff'}
    pal2 = {2:'#800000',1:'#ff8000',0:'#003366'}

    for i, row in df.iterrows():
        lm.ax.plot(row['roi'], row['age'], ms=5.5, mew=0.4, marker='o', ls='-', mfc=pal[row['apoe']], mec=pal2[row['apoe']], lw=1)
    #lm.savefig('/tmp/fig.svg')
    lm.fig.suptitle(roi_name)

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

def lm_plot2(data, roi_name):
    import seaborn as sns
    roi_name = data.columns[1]
    df = pd.DataFrame(data, columns=['apoe','age','centiloids', 'gender', roi_name]).rename(columns={roi_name:'roi'})
    df = df.dropna()
    lm = sns.lmplot(x='age', y='roi',  data=df, size=6.2, hue='centiloids', aspect=1.35, ci=90, legend=False,
	 truncate=True, sharex=False,sharey=False)
    ax = lm.axes
    ax[0,0].set_ylim([df['roi'].min(), df['roi'].max()])
    ax[0,0].set_xlim([50,70])
    ax[0,0].set_yticklabels(['%.2e'%x for x in ax[0,0].get_yticks()])
    ax[0,0].tick_params(labelsize=12)
    ax[0,0].set_ylabel('')
    ax[0,0].set_xlabel('age', fontsize=15, weight='bold')
    lm.ax.plot(df['roi'], df['age'], ms=5.5, mew=0.4, marker='o', ls='-', lw=1)
    lm.fig.suptitle(roi_name)
