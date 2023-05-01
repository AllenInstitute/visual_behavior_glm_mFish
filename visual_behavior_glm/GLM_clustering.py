import numpy as np
import pandas as pd
import pickle
from scipy.stats import chisquare
from scipy.stats import chi2_contingency
from scipy.stats import power_divergence
from scipy.stats import fisher_exact
# import FisherExact (Used for non2x2 tables of Fisher Exact test, not used but leaving a note)
import matplotlib.pyplot as plt
# import visual_behavior.data_access.loading as loading
from mpl_toolkits.axes_grid1 import make_axes_locatable
from brain_observatory_analysis.utilities import image_utils as utils

def get_clustering_dir(version='01_nonridgit_events'):

    filedir = f'//allen/programs/braintv/workgroups/nc-ophys/omFish_glm/ophys_glm/v_{version}/figures/clustering/'
    return filedir

def compare_stats(num_shuffles=1000):
    plt.figure()
    for i in np.arange(100,1000,10):
        stats = compare_shuffle(n=i,num_shuffles=num_shuffles)
        pval = stats[0]
        chi = stats[1]
        if i==100:
            plt.plot(i,chi,'bo',label='chi-2')
            plt.plot(i,pval,'ro',label='distribution')
        else:
            plt.plot(i,chi,'bo')
            plt.plot(i,pval,'ro')

    plt.axhline(.05, linestyle='--',color='k')
    plt.legend()
    plt.ylabel('p value')
    plt.xlabel('num cells')
    plt.ylim(0,.5)

def compare_shuffle(n=100,p=.15,pn=.1,num_shuffles=1000):  
    # worried about independence
    num_h = int(np.floor(pn*n))
    num_m = n-num_h
    raw = [1]*(num_h)+[0]*(num_m)
    
    # Generate shuffle 
    num_hits =[]
    for i in np.arange(0,num_shuffles):
        #shuffle = np.random.rand(n) < pn
        shuffle = np.random.choice(raw,n)
        num_hits.append(np.sum(shuffle))

    # Compute chi-square where the data is 15%/85% of cells
    # and null is the mean % across shuffles
    data = [p*n*1000000000, (1-p)*n*1000000000]
    null = [np.mean(num_hits), n-np.mean(num_hits)]
    x = np.floor(np.array([data,null])).T
    out = chi2_contingency(x,correction=True)

    # Compare that with a p-value where we ask what percentage of the shuffles 
    # had more than 15%
    pval = np.sum(np.array(num_hits) >= p*n)/num_shuffles*2 

    return pval,out[1]


def final(df, cre,areas=None,test='chi_squared_'):
    '''
        Returns two tables
        proportion_table contains the proportion of cells in each location found in each cluster, relative to the average proportion across location for that cluster

        stats_table returns statistical tests on the proportion of cells in each location 
        Use 'bh_significant' unless you have a good reason to use the uncorrected tests

        areas should be a list of the locations to look at.
    '''
    if areas is None:
        # Get areas
        areas = np.sort(df['location'].unique())   
    else:
        assert set(areas) == set(df['location'].unique()), "areas passed in don't match location column" 

    proportion_table = compute_cluster_proportion_cre(df, cre,areas)
    stats_table = stats(df,cre,areas,test=test)
    return proportion_table, stats_table

def cluster_frequencies():
    '''
        Generates 4 different plots of clustering frequency/proportion analysis
        1. The proportions of each location in each cluster
        2. The proportions of each location in each cluster 
           relative to "chance" of 1/n-clusters (evenly distributed cells across clusters)
        3. The proportions of each location in each cluster
           relative to the average proportion across locations in that cluster 
           (clusters have the same proportion across locations)
        4. The proportion of each location in each cluster
           relative to the average proportion across locations in that cluster
           but using a multiplicative perspective instead of a linear perspective. 
    '''
    df = load_cluster_labels()
    plot_proportions(df)
    plot_proportion_differences(df)
    plot_cluster_proportions(df)
    plot_cluster_percentages(df)   
 
def load_cluster_labels():
    '''
        - Loads a dataframe of cluster labels
        - merges in cell table data 
        - defines a `location` column with depth/area combinations
        - drops clusters with less than 5 cells
    '''

    # Load cluster labels
    filepath = '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_plots/figure_4/24_events_all_L2_optimize_by_session/220622_across_session_norm_10_5_10/cluster_labels_Vip_10_Sst_5_Slc17a7_10.h5'
    df = pd.read_hdf(filepath, key='df')

    # load cell data
    cells_table = loading.get_cell_table(platform_paper_only=True,include_4x2_data=False).reset_index().drop_duplicates(subset='cell_specimen_id')
    df = df.drop(columns=['labels','cre_line'])
    df = pd.merge(df, cells_table, on='cell_specimen_id',suffixes=('','_y'))  

    # Bin depths and annotate
    df['coarse_binned_depth'] = ['upper' if x < 250 else 'lower' for x in df['imaging_depth']]
    df['location'] = df['targeted_structure']+'_'+df['coarse_binned_depth']

    # Remove clusters with less than 5 cells
    #df = df.drop(df.index[(df['cre_line']=="Sst-IRES-Cre")&(df['cluster_id']==6)])
    #df = df.drop(df.index[(df['cre_line']=="Slc17a7-IRES2-Cre")&(df['cluster_id']==10)])

    return df

def plot_proportions(df,areas=None,savefig=False,extra='',test='chi_squared_'):
    '''
        Compute, then plot, the proportion of cells in each location within each cluster
    '''
    
    if areas is None:
        # Get areas
        areas = np.sort(df['location'].unique())   
    else:
        assert set(areas) == set(df['location'].unique()), "areas passed in don't match location column" 

    fig, ax = plt.subplots(1,3,figsize=(8,4))
    fig.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9, wspace=1)
    plot_proportion_cre(df,areas, fig, ax[2], 'Gad2-IRES-Cre',test=test)
    if savefig:
        extra = extra+'_'+test
        plt.savefig(get_clustering_dir()+'cluster_proportions'+extra+'.svg')
        plt.savefig(get_clustering_dir()+'cluster_proportions'+extra+'.png')

def compute_proportion_cre(df, cre,areas):
    '''
        Computes the proportion of cells in each cluster within each location
        
        location must be a column of string names. The statistics are computed relative to whatever the location column contains
    '''

    # Count cells in each area/cluster
    table = df.query('cre_line == @cre').groupby(['cluster_id','location'])['cell_specimen_id'].count().unstack()
    table = table[areas]
    table = table.fillna(value=0)

    # compute fraction in each area/cluster
    for a in areas:
        table[a] = table[a]/table[a].sum()
    return table

def plot_proportion_cre(df,areas, fig,ax, cre,test='chi_squared_'):
    '''
        Fraction of cells per area&depth 
    '''

    # Get proportions
    table = compute_proportion_cre(df, cre,areas)

    # plot proportions
    cbar = ax.imshow(table,cmap='Purples',vmax=.4)
    ax.set_ylabel('Cluster #',fontsize=16)  
    ax.set_yticks(range(0,len(table)))
    ax.set_yticklabels(range(1,len(table)+1))
    ax.set_title(mapper(cre),fontsize=16) 
    fig.colorbar(cbar, ax=ax,label='fraction of cells per location')
   
    # Add statistics 
    table2 = stats(df, cre,areas,test=test)
    for index in table2.index.values:
        if table2.loc[index]['bh_significant']:
            ax.plot(-1,index-1,'r*')

    num_areas = len(areas)
    ax.set_xlim(-1.5,num_areas-.5)
    ax.set_xticks(range(-1,num_areas))
    ax.set_xticklabels(np.concatenate([[test[:-1]],areas]),rotation=90)
    ax.axvline(-0.5,color='k',linewidth=.5)

def plot_proportion_differences(df,areas=None):
    '''
        Computes, then plots, the proportion of cells in each location within each cluster
        relative to a 1/n average distribution across n clusters. 
    '''

    if areas is None:
        # Get areas
        areas = np.sort(df['location'].unique())    
    else:
        assert set(areas) == set(df['location'].unique()), "areas passed in don't match location column" 

    fig, ax = plt.subplots(1,3,figsize=(8,4))
    fig.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9, wspace=1)
    plot_proportion_differences_cre(df, areas, fig, ax[2], 'Slc17a7-IRES2-Cre')
    plot_proportion_differences_cre(df, areas, fig, ax[1], 'Sst-IRES-Cre')
    plot_proportion_differences_cre(df, areas, fig, ax[0], 'Vip-IRES-Cre')
    plt.savefig(get_clustering_dir()+'cluster_proportion_differences.svg')
    plt.savefig(get_clustering_dir()+'cluster_proportion_differences.png')

def compute_proportion_differences_cre(df, cre,areas):
    '''
        compute proportion differences relative to 1/n average
    '''
    # count cells in each area/cluster
    table = df.query('cre_line == @cre').groupby(['cluster_id','location'])['cell_specimen_id'].count().unstack()
    table = table[areas]
    table = table.fillna(value=0)
    nclusters = len(table.index.values)

    # compute fraction in each area relative to expected fraction 
    for a in areas:
        table[a] = table[a]/table[a].sum() - 1/nclusters
    return table

def plot_proportion_differences_cre(df,areas, fig,ax, cre):
    '''
        Fraction of cells per location, then
        subtract expected fraction (1/n)
    '''
    table = compute_proportion_differences_cre(df,cre,areas)

    # plot fractions
    vmax = table.abs().max().max()
    cbar = ax.imshow(table,cmap='PRGn',vmin=-vmax,vmax=vmax)
    fig.colorbar(cbar, ax=ax,label='fraction of cells per location \nrelative to evenly distributed across clusters')
    ax.set_xticks(range(0,len(areas)))
    ax.set_xticklabels(areas,rotation=90)
    ax.set_ylabel('Cluster #',fontsize=16)  
    ax.set_title(mapper(cre),fontsize=16) 

def plot_cluster_proportions(df,areas=None):
    if areas is None:
        # Get areas
        areas = np.sort(df['location'].unique())    
    else:
        assert set(areas) == set(df['location'].unique()), "areas passed in don't match location column" 

    fig, ax = plt.subplots(1,3,figsize=(14,8))
    fig.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9, wspace=1)
    plot_cluster_proportion_cre(df,areas, fig, ax[2], 'Slc17a7-IRES2-Cre')
    plot_cluster_proportion_cre(df,areas, fig, ax[1], 'Sst-IRES-Cre')
    plot_cluster_proportion_cre(df,areas, fig, ax[0], 'Vip-IRES-Cre')
    plt.savefig(get_clustering_dir()+'within_cluster_proportions.svg')
    plt.savefig(get_clustering_dir()+'within_cluster_proportions.png')
    plt.savefig(get_clustering_dir() + 'within_cluster_proportions.pdf')

def compute_cluster_proportion_cre(df, cre,areas):
    table = compute_proportion_cre(df, cre,areas)

    # get average proportion in each cluster
    table['mean'] = table.mean(axis=1)

    # compute proportion in each area relative to cluster average
    for a in areas:
        table[a] = table[a] - table['mean'] 

    # plot proportions
    table = table[areas]
    return table

def plot_cluster_proportion_cre(df,areas,fig,ax, cre,test='chi_squared_'):
    '''
        Fraction of cells per area&depth 
    '''
    table = compute_cluster_proportion_cre(df,cre,areas)

    vmax = table.abs().max().max()
    vmax = .15
    cbar = ax.imshow(table,cmap='PRGn',vmin=-vmax, vmax=vmax)
    fig.colorbar(cbar, ax=ax,label='proportion of cells per location \n relative to cluster average')
    ax.set_xticks(range(0,len(areas)))
    ax.set_xticklabels(areas,rotation=90)
    ax.set_ylabel('Cluster #',fontsize=16)  
    ax.set_title(mapper(cre),fontsize=16)
    ax.set_yticks(np.arange(len(table)))
    ax.set_yticklabels(np.arange(1,len(table)+1))
    
    # add statistics
    table2 = stats(df, cre,areas,test=test)
    for index in table2.index.values:
        if table2.loc[index]['bh_significant']:
            ax.plot(-1,index-1,'r*')
    ax.set_xlim(-1.5,len(areas)-.5)
    ax.set_xticks(range(-1,len(areas)))
    ax.set_xticklabels(np.concatenate([['p<0.05'],areas]),rotation=90)
    ax.axvline(-0.5,color='k',linewidth=.5)

def plot_cluster_percentages(df,areas=None):
    if areas is None:
        # Get areas
        areas = np.sort(df['location'].unique())    
    else:
        assert set(areas) == set(df['location'].unique()), "areas passed in don't match location column" 

    fig, ax = plt.subplots(1,3,figsize=(8,4))
    fig.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9, wspace=1)
    plot_cluster_percentage_cre(df,areas, fig, ax[2], 'Slc17a7-IRES2-Cre')
    plot_cluster_percentage_cre(df,areas, fig, ax[1], 'Sst-IRES-Cre')
    plot_cluster_percentage_cre(df,areas, fig, ax[0], 'Vip-IRES-Cre')
    plt.savefig(get_clustering_dir()+'within_cluster_percentages.svg')
    plt.savefig(get_clustering_dir()+'within_cluster_percentages.png')

def plot_cluster_percentage_cre(df,areas,fig,ax, cre,test='chi_squared_'):
    '''
        Fraction of cells per area&depth 
    '''
    # count cells in each area/cluster
    table = df.query('cre_line == @cre').groupby(['cluster_id','location'])['cell_specimen_id'].count().unstack()
    table = table[areas]
    table = table.fillna(value=0)

    # compute proportion in each area/cluster
    for a in areas:
        table[a] = table[a]/table[a].sum()

    # get average proportion in each cluster
    table['mean'] = table.mean(axis=1)

    # build second table with cells in each area/cluster
    table2 = df.query('cre_line == @cre').groupby(['cluster_id','location'])['cell_specimen_id'].count().unstack()
    table2 = table2[areas]
    table2 = table2.fillna(value=0)

    # estimate chance cell counts based on average proportion in each cluster
    # then add relative fraction of actual counts compared to chance counts
    # subtract 1 so 0=chance
    for a in areas:
        table2[a+'_chance_count'] = table2[a].sum()*table['mean']
        table2[a+'_rel_fraction'] = table2[a]/table2[a+'_chance_count']-1
   
    area_rel_fraction = [area+'_rel_fraction' for area in areas]
 
    # plot proportions 
    table2 = table2[area_rel_fraction]
    cbar = ax.imshow(table2,cmap='PRGn',vmin=-1,vmax=1)
    fig.colorbar(cbar, ax=ax,label='fraction of cells per area & depth')
    ax.set_ylabel('Cluster #',fontsize=16)  
    ax.set_title(mapper(cre),fontsize=16)
    
    # add statistics
    table2 = stats(df, cre,areas,test=test)
    for index in table2.index.values:
        if table2.loc[index]['bh_significant']:
            ax.plot(-1,index-1,'r*')
    ax.set_xlim(-1.5,len(areas)-.5)
    ax.set_xticks(range(-1,len(areas)))
    ax.set_xticklabels(np.concatenate([['p<0.05'],areas]),rotation=90)
    ax.axvline(-0.5,color='k',linewidth=.5)

def stats(df,cre,areas,test='chi_squared_',lambda_str='log-likelihood'):
    '''
        Performs chi-squared tests to asses whether the observed cell counts 
        in each area/depth differ significantly from the average for that cluster. 
    '''    

    # compute cell counts in each area/cluster
    table = df.query('cre_line == @cre').\
        groupby(['cluster_id','location'])['cell_specimen_id'].count().unstack()
    table = table[areas]
    table = table.fillna(value=0)

    # compute proportion for null hypothesis that areas have the same proportions
    table['total_cells'] = table.sum(axis=1)
    table['null_mean_proportion'] = table['total_cells']/np.sum(table['total_cells'])

    # second table of cell counts in each area/cluster
    table2 = df.query('cre_line == @cre').\
        groupby(['cluster_id','location'])['cell_specimen_id'].count().unstack()
    table2 = table2[areas]
    table2 = table2.fillna(value=0)

    # compute estimated frequency of cells based on average fraction for each cluster
    for a in areas:
        table2[a+'_chance_count'] = table2[a].sum()*table['null_mean_proportion']

    # perform chi-squared test
    area_chance = [area+'_chance_count' for area in areas]
    for index in table2.index.values:
        f = table2.loc[index][areas].values
        f_expected = table2.loc[index][area_chance].values
        not_f = table2[areas].sum().values - f       
 
        # Manually doing check here bc Im on old version of scipy
        assert np.abs(np.sum(f) - np.sum(f_expected))<1, \
            'f and f_expected must be the same'

        if test == 'chi_squared_':
            out = chisquare(f,f_expected)
            table2.at[index, test+'pvalue'] = out.pvalue
            table2.at[index, 'significant'] = out.pvalue < 0.05
        elif test == 'g_test_':
            f = f.astype(np.double)
            f_expected = f_expected.astype(np.double)
            out = power_divergence(f, f_expected,lambda_=lambda_str)
            table2.at[index, test+'pvalue'] = out.pvalue
            table2.at[index, 'significant'] = out.pvalue < 0.05           
        elif test == 'fisher_':
            contingency = np.array([f,not_f]) 
            if np.shape(contingency)[1] > 2:
                raise Exception('Need to import FisherExact package for non 2x2 tables')
                #pvalue = FisherExact.fisher_exact(contingency)
            else:
                oddsratio, pvalue = fisher_exact(contingency)
            table2.at[index, test+'pvalue'] = pvalue
            table2.at[index, 'significant'] = pvalue < 0.05              

    # Use Benjamini Hochberg Correction for multiple comparisons
    table2 = add_hochberg_correction(table2,test=test) 
    return table2

def mapper(cre):
    mapper = {
        'Slc17a7-IRES2-Cre':'Excitatory',
        'Sst-IRES-Cre':'Sst',
        'Vip-IRES-Cre':'Vip',
        'Gad2-IREB2-Cre':'Inhibitory',}
    return mapper[cre]

def add_hochberg_correction(table,test='chi_squared_'):
    '''
        Performs the Benjamini Hochberg correction
    '''    
    # Sort table by pvalues
    table = table.sort_values(by=test+'pvalue').reset_index()
    
    # compute the corrected pvalue based on the rank of each test
    # Need to use rank starting at 1
    table['imq'] = (1+table.index.values)/len(table)*0.05

    # Find the largest pvalue less than its corrected pvalue
    # all tests above that are significant
    table['bh_significant'] = False
    passing_tests = table[table[test+'pvalue'] < table['imq']]
    if len(passing_tests) >0:
        last_index = table[table[test+'pvalue'] < table['imq']].tail(1).index.values[0]
        table.at[last_index,'bh_significant'] = True
        table.at[0:last_index,'bh_significant'] = True
    
    # reset order of table and return
    return table.sort_values(by='cluster_id').set_index('cluster_id') 

 ## Code below has been moved from VBA.dimentionality_reduction.clustering.plotting

from brain_observatory_analysis.utilities import image_utils as utils
import seaborn as sns

def get_abbreviated_experience_levels(experience_levels):
    """
    converts experience level names (ex: 'Novel >1') into short hand versions (ex: 'N>1')
    abbreviated names are returned in the same order as provided in experience_levels
    """
    exp_level_abbreviations = [exp_level.split(' ')[0][0] if len(exp_level.split(' ')) == 1 else exp_level.split(' ')[0][0] + exp_level.split(' ')[1][:2] for exp_level in experience_levels]
    # exp_level_abbreviations = ['F', 'N', 'N+']
    return exp_level_abbreviations

def get_abbreviated_features(features):
    """
    converts GLM feature names (ex: 'Behavioral') into first letter capitalized short hand versions (ex: 'B')
    'all-images' gets converted to 'I' for Images
    abbreviated names are returned in the same order as provided in features
    """
    # get first letter of each feature name and capitalize it
    feature_abbreviations = [feature[0].upper() for feature in features]
    # change the "A" for "all-images" to "I" for "images" instead
    if 'A' in feature_abbreviations:
        feature_abbreviations[feature_abbreviations.index('A')] = 'I'
    return feature_abbreviations

def get_cre_lines(cluster_labels):
    """
    get list of cre lines in cell_metadata and sort in reverse order so that Vip is first
    cell_metadata is a table similar to ophys_cells_table from SDK but can have additional columns based on clustering results
    """
    cre_lines = np.sort(cluster_labels.cre_line.unique())
    return cre_lines

def get_feature_matrix_for_cre_line(feature_matrix, cell_metadata, cre_line, dropna=True):
    """
    limit feature_matrix to cell_specimen_ids for this cre line
    """
    cre_cell_specimen_ids = cell_metadata[cell_metadata['cre_line'] == cre_line].index.values
    feature_matrix_cre = feature_matrix.loc[cre_cell_specimen_ids].copy()
    if dropna is True:
        feature_matrix_cre = feature_matrix_cre.dropna(axis=0)
    return feature_matrix_cre

def plot_feature_matrix_for_cre_lines(feature_matrix, cluster_labels, use_abbreviated_labels=False, save_dir=get_clustering_dir(), folder=''):
    """
    plots the feature matrix used for clustering where feature matrix consists of cell_specimen_ids as rows,
    and features x experience levels as columns, for cells matched across experience levels
    :param feature_matrix:
    :param cluster_labels: table with cell_specimen_id as index and metadata as columns
    :param save_dir: directory to save plot
    :param folder:
    :return:
    """
    # check if there are negative values in feature_matrix, if so, use diff cmap and set vmin to -1
    if len(np.where(feature_matrix < 0)[0]) > 0:
        vmin = -1
        cmap = 'RdBu'
    else:
        vmin = 0
        cmap = 'Blues'
    
    cre_lines = get_cre_lines(cluster_labels)
    n_cre_lines = len(cre_lines)

    figsize = (5*n_cre_lines, 7)
    fig, axes = plt.subplots(1, n_cre_lines, figsize=figsize)
    for i, cre_line in enumerate(cre_lines):
        if n_cre_lines == 1:
            ax=axes
        else:
            ax=axes[i]
        # data = get_feature_matrix_for_cre_line(feature_matrix, cluster_labels, cre_line)
        data=feature_matrix.copy()
        ax = sns.heatmap(data.values, cmap=cmap, ax=ax, vmin=vmin, vmax=1,
                            robust=True, cbar_kws={"drawedges": False, "shrink": 0.7, "label": 'coding score'})
        for x in [3, 6, 9]:
            ax.axvline(x=x, ymin=0, ymax=data.shape[0], color='gray', linestyle='--', linewidth=1)
        ax.set_title(cre_line, fontsize=16)
        ax.set_ylabel('cells')
        ax.set_ylim(0, data.shape[0])
        ax.set_yticks([0, data.shape[0]])
        ax.set_yticklabels((0, data.shape[0]), fontsize=14)
        ax.set_ylim(ax.get_ylim()[::-1])  # flip y axes so larger clusters are on top
        ax.set_xlabel('')
        ax.set_xlim(0, data.shape[1])
        ax.set_xticks(np.arange(0, data.shape[1]) + 0.5)
        if use_abbreviated_labels:
            xticklabels = [get_abbreviated_experience_levels([key[1]])[0] + ' -  ' + get_abbreviated_features([key[0]])[0].upper() for key in list(data.keys())]
            ax.set_xticklabels(xticklabels, rotation=90, fontsize=14)
        else:
            ax.set_xticklabels([key[1] + ' -  ' + key[0] for key in list(data.keys())], rotation=90, fontsize=14)

    fig.subplots_adjust(wspace=0.7)

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'feature_matrix_unsorted')



def plot_feature_matrix_sorted(feature_matrix, cluster_labels, sort_col='cluster_id', use_abbreviated_labels=False,
                               save_dir=get_clustering_dir(), folder='', suffix=''):
    """
    plots feature matrix used for clustering sorted by sort_col

    sort_col: column in cluster_labels to sort rows of feature_matrix (cells) by
    """
    # check if there are negative values in feature_matrix, if so, use diff cmap and set vmin to -1
    if len(np.where(feature_matrix < 0)[0]) > 0:
        vmin = -1
        cmap = 'RdBu'
    else:
        vmin = 0
        cmap = 'Blues'
    cre_lines = get_cre_lines(cluster_labels)
    n_cre_lines = len(cre_lines)
    n_clusters = len(cluster_labels.cluster_id.unique())

    figsize = (5*n_cre_lines, 7)
    fig, axes = plt.subplots(1, n_cre_lines, figsize=figsize)

    for i, cre_line in enumerate(cre_lines):
        # get cell ids for this cre line in sorted order
        if n_cre_lines == 1:
            ax=axes
        else:
            ax=axes[i]
        print(cre_line)
        sorted_cluster_meta_cre = cluster_labels[cluster_labels.cre_line == cre_line].sort_values(by=sort_col)
        cell_order = sorted_cluster_meta_cre['cell_specimen_id'].values
        label_values = sorted_cluster_meta_cre[sort_col].values

        # get data from feature matrix for this set of cells
        data = feature_matrix.loc[cell_order]
        ax = sns.heatmap(data.values, cmap=cmap, ax=ax, vmin=vmin, vmax=1,
                            robust=True, cbar_kws={"drawedges": False, "shrink": 0.7, "label": 'coding score'})

        for x in [3, 6, 9]:
            ax.axvline(x=x, ymin=0, ymax=data.shape[0], color='gray', linestyle='--', linewidth=1)
        #  ax.set_title(get_cell_type_for_cre_line(cre_line, cluster_meta))
        ax.set_title(cre_line, fontsize=16)
        ax.set_ylabel('cells')
        ax.set_ylim(0, data.shape[0])
        ax.set_yticks([0, data.shape[0]])
        ax.set_yticklabels((0, data.shape[0]), fontsize=14)
        ax.set_ylim(ax.get_ylim()[::-1])  # flip y axes so larger clusters are on top
        ax.set_xlabel('')
        ax.set_xlim(0, data.shape[1])
        ax.set_xticks(np.arange(0, data.shape[1]) + 0.5)
        if use_abbreviated_labels:
            xticklabels = [get_abbreviated_experience_levels([key[1]])[0] + ' -  ' + get_abbreviated_features([key[0]])[0].upper() for key in list(data.keys())]
            ax.set_xticklabels(xticklabels, rotation=90, fontsize=14)
        else:
            ax.set_xticklabels([key[1] + ' -  ' + key[0] for key in list(data.keys())], rotation=90, fontsize=14)

        # plot a line at the division point between clusters
        cluster_divisions = np.where(np.diff(label_values) == 1)[0]
        for y in cluster_divisions:
            ax.hlines(y, xmin=0, xmax=data.shape[1], color='k')

    fig.subplots_adjust(wspace=0.7)
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, f'{n_clusters}_feature_matrix_sorted_by_' + sort_col + suffix)

def plot_dropout_heatmap(cluster_meta, feature_matrix, cluster_id, cbar=False,
                         abbreviate_features=False, abbreviate_experience=False,
                         cluster_size_in_title=True, small_fontsize=False, ax=None):

    # check if there are negative values in feature_matrix, if so, use diff cmap and set vmin to -1
    if len(np.where(feature_matrix < 0)[0]) > 0:
        vmin = -1
        cmap = 'RdBu'
    else:
        vmin = 0
        cmap = 'Blues'
    cre_csids = cluster_meta.index.values
    this_cluster_meta = cluster_meta[(cluster_meta['cluster_id'] == cluster_id)]
    this_cluster_csids = this_cluster_meta['cell_specimen_id'].values
    mean_dropout_df = feature_matrix.loc[this_cluster_csids].mean().unstack()
    features = ['all-images', 'behavioral','omissions',  'task']
    #mean_dropout_df = mean_dropout_df.loc[features]  # order regressors in a specific order

    if ax is None:
        fig, ax = plt.subplots()
    ax = sns.heatmap(mean_dropout_df, cmap=cmap, vmin=vmin, vmax=1, ax=ax, cbar=cbar, cbar_kws={'label': 'coding score'})
    if cluster_size_in_title:
        # fraction is number of cells in this cluster vs all cells in this cre line
        fraction_cre = len(this_cluster_csids) / float(len(cre_csids))
        fraction = np.round(fraction_cre * 100, 1)
        # set title and labels
        ax.set_title('cluster ' + str(cluster_id) + '\n' + str(fraction) + '%, n=' + str(len(this_cluster_csids)))
    else:
        # title is cre line abbreviation and cluster #
        cell_type = 'inhibitory'
        ax.set_title(cell_type + ' cluster ' + str(cluster_id))
    ylabels=['images', 'omissions', 'behavioral',  'task']
    ax.set_yticks(np.arange(0.5, len(mean_dropout_df.index.values) + 0.5))
    ax.set_yticklabels(ylabels, rotation=0, fontsize=14)
    # if abbreviate_features:
    #     # set yticks to abbreviated feature labels
    #     feature_abbreviations = get_abbreviated_features(mean_dropout_df.index.values)
    #     ax.set_yticklabels(feature_abbreviations, rotation=0)
    # else:
    # if abbreviate_experience:
    #     # set xticks to abbreviated experience level labels
    #     exp_level_abbreviations = get_abbreviated_experience_levels(mean_dropout_df.columns.values)
    #     ax.set_xticklabels(exp_level_abbreviations, rotation=90)
    # else:
    xlabels = ['Familiar', 'Novel', 'Novel +']
    ax.set_xticklabels(xlabels, rotation=90, fontsize=14)
    ax.set_ylim(0, mean_dropout_df.shape[0])
    # invert y axis so images is always on top
    ax.invert_yaxis()
    ax.set_xlabel('')
    return ax


def plot_clusters_row(cluster_meta, feature_matrix, save_fig=True, tag=''):
    filedir = get_clustering_dir()

    cluster_ids = np.sort(cluster_meta.cluster_id.unique())
    # if order to sort clusters is provided, use it

    # cluster_remap = {}
    # for new_id, old_id in enumerate(sort_order[cre_line]):
    #     cluster_remap[old_id] = new_id + 1
    # cluster_meta['cluster_id'].replace(cluster_remap, inplace=True)

    n_clusters = len(cluster_ids)

    n_rows = 1
    figsize = (n_clusters * 3.5, n_rows * 2)
    fig, ax = plt.subplots(n_rows, n_clusters, figsize=figsize, sharex='row', sharey='row')
    # gridspec_kw={'height_ratios': [1, 0.75]})
    ax = ax.ravel()
    for i, cluster_id in enumerate(cluster_ids):
        # plot mean dropout heatmap for this cluster
        ax[i] = plot_dropout_heatmap(cluster_meta, feature_matrix, cluster_id,ax=ax[i])


    fig.subplots_adjust(hspace=1.2, wspace=0.6)
    #plt.title(tag)
    fig.suptitle(tag, x=0.52, y=1.3, fontsize=16)
    plt.tight_layout()
    if save_fig:
        utils.save_figure(fig, figsize, filedir, '', f'mean_clusters_{n_clusters}_'+tag)


def apply_roi_classifier(cell_table, filter_col = 'iscell_p0.2', filter_val=0):
    
    '''Laods results of a linear classifier, applying invalid roi filter to provided df, removes cell specimen ids associated with invalid rois'''

    filename = '//allen/programs/mindscope/workgroups/learning/pipeline_validation/classify_rois_sac2023/'\
               'LAMF_NR_SAC2023_iscell_lr_model_version2.pkl'
    with open(filename, 'rb') as f:
        cl_table = pickle.load(f)
        f.close()

    unfiltered_rois = cell_table.cell_roi_id.unique()
    invalid_rois = cl_table[cl_table[filter_col]<=filter_val]['cell_roi_id'].values
    matched_invalid_rois = np.intersect1d(invalid_rois, unfiltered_rois)
    print(f'found {len(matched_invalid_rois)} invalid rois in the dataframe')
    # get cell_specimn_ids with those invalid rois
    invalid_cell_specimen_ids = cell_table[cell_table.cell_roi_id.isin(matched_invalid_rois)]['cell_specimen_id'].unique()
    print(f'removing {len(invalid_cell_specimen_ids)} cell specimen ids')
    cell_table_filtered = cell_table[cell_table.cell_specimen_id.isin(invalid_cell_specimen_ids)==False]
    
    return cell_table_filtered


