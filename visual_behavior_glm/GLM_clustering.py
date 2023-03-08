import pickle
import os
import numpy as np
import pandas as pd
from scipy.stats import chisquare
from scipy.stats import chi2_contingency
from scipy.stats import power_divergence
from scipy.stats import fisher_exact
from scipy.stats import sem
from scipy.sparse import csgraph

# for clustering
from sklearn.metrics import pairwise_distances
from sklearn.cluster import SpectralClustering

# import FisherExact (Used for non2x2 tables of Fisher Exact test, not used but leaving a note)
import matplotlib.pyplot as plt
import visual_behavior.data_access.loading as loading
from mpl_toolkits.axes_grid1 import make_axes_locatable

version = 'v_testing_01'
folder = 'clustering'
filedir = '//allen/programs/braintv/workgroups/nc-ophys/omFish_glm/ophys_glm/'+version+'/'+folder+'/'

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
    plot_proportion_cre(df,areas, fig, ax[2], 'Slc17a7-IRES2-Cre',test=test)
    plot_proportion_cre(df,areas,  fig, ax[1], 'Sst-IRES-Cre',test=test)
    plot_proportion_cre(df,areas,  fig, ax[0], 'Vip-IRES-Cre',test=test)
    if savefig:
        extra = extra+'_'+test
        plt.savefig(filedir+'cluster_proportions'+extra+'.svg')
        plt.savefig(filedir+'cluster_proportions'+extra+'.png')

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
    plt.savefig(filedir+'cluster_proportion_differences.svg')
    plt.savefig(filedir+'cluster_proportion_differences.png')

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
    plt.savefig(filedir+'within_cluster_proportions.svg')
    plt.savefig(filedir+'within_cluster_proportions.png')
    plt.savefig(filedir + 'within_cluster_proportions.pdf')

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
    plt.savefig(filedir+'within_cluster_percentages.svg')
    plt.savefig(filedir+'within_cluster_percentages.png')

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
        'Gad2-IRES-Cre': 'Inhibitory',
        'Rbp4-Cre_KL100': 'Rbp4',
        'Cux2-CreERT2': 'Cux2',
        }
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

    
def shuffle_dropout_score(feature_matrix, shuffle_type='all'):
    '''
    Shuffles dataframe with dropout scores from GLM.
    shuffle_type: str, default='all', other options= 'experience', 'regressors', 'experience_within_cell',
                        'full_experience'

    Returns:
        df_shuffled (pd. Dataframe) of shuffled dropout scores
    '''
    df_shuffled = feature_matrix.copy()
    regressors = feature_matrix.columns.levels[0].values
    experience_levels = feature_matrix.columns.levels[1].values
    if shuffle_type == 'all':
        print('shuffling all data')
        for column in feature_matrix.columns:
            df_shuffled[column] = feature_matrix[column].sample(frac=1).values

    elif shuffle_type == 'experience':
        print('shuffling data across experience')
        assert np.shape(feature_matrix.columns.levels)[
            0] == 2, 'df should have two level column structure, 1 - regressors, 2 - experience'
        for experience_level in experience_levels:
            randomized_cids = feature_matrix.sample(frac=1).index.values
            for i, cid in enumerate(randomized_cids):
                for regressor in regressors:
                    df_shuffled.iloc[i][(regressor, experience_level)] = feature_matrix.loc[cid][(regressor, experience_level)]

    elif shuffle_type == 'regressors':
        print('shuffling data across regressors')
        assert np.shape(feature_matrix.columns.levels)[
            0] == 2, 'df should have two level column structure, 1 - regressors, 2 - experience'
        for regressor in regressors:
            randomized_cids = feature_matrix.sample(frac=1).index.values
            for i, cid in enumerate(randomized_cids):
                for experience_level in experience_levels:
                    df_shuffled.iloc[i][(regressor, experience_level)] = feature_matrix.loc[cid][(regressor, experience_level)]

    elif shuffle_type == 'experience_within_cell':
        print('shuffling data across experience within each cell')
        cids = feature_matrix.index.values
        experience_level_shuffled = experience_levels.copy()
        for cid in cids:
            np.random.shuffle(experience_level_shuffled)
            for j, experience_level in enumerate(experience_level_shuffled):
                for regressor in regressors:
                    df_shuffled.loc[cid][(regressor, experience_levels[j])] = feature_matrix.loc[cid][(regressor,
                                                                                                   experience_level)]
    elif shuffle_type == 'full_experience':
        print('shuffling data across experience fully (cell id and experience level)')
        assert np.shape(feature_matrix.columns.levels)[
            0] == 2, 'df should have two level column structure, 1 - regressors, 2 - experience'
        # Shuffle cell ids first
        for experience_level in experience_levels:
            randomized_cids = feature_matrix.sample(frac=1).index.values
            for i, cid in enumerate(randomized_cids):
                for regressor in regressors:
                    df_shuffled.iloc[i][(regressor, experience_level)] = feature_matrix.loc[cid][
                        (regressor, experience_level)]
        # Shuffle experience labels
        df_shuffled_again = df_shuffled.copy(deep=True)
        cids = df_shuffled.index.values
        experience_level_shuffled = experience_levels.copy()
        for cid in cids:
            np.random.shuffle(experience_level_shuffled)
            for j, experience_level in enumerate(experience_level_shuffled):
                for regressor in regressors:
                    df_shuffled_again.loc[cid][(regressor, experience_levels[j])] = df_shuffled.loc[cid][(regressor,
                                                                                                          experience_level)]

        df_shuffled = df_shuffled_again.copy()

    else:
        print('no such shuffle type..')
        df_shuffled = None
    return df_shuffled
 

# Funcltions below were moved from visual behavior analysis repo

def save_clustering_results(data, filename_string='', path=None):
    '''
    for HCP scripts to save output of spectral clustering in a specific folder
    input:
        data: what to save
        filename_string: name of the file, use as descriptive info as possible
        path: where to save the file, default is the files folder in the same directory as this file

    return:
        nothing, just saves the file
    '''
    if path is None:
        path = filedir+'/files'
    if os.path.exists(path) is False:
        os.mkdir(path)

    filename = os.path.join(path, '{}'.format(filename_string))
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    f.close()

def get_features_for_clustering():
    """
    get GLM features to use for clustering analysis
    """
    features = ['all-images', 'omissions', 'behavioral', 'task']
    return features

def get_cell_metadata_for_feature_matrix(feature_matrix, cells_table):
    """
    get a dataframe of cell metadata for all cells in feature_matrix
    limits ophys_cells_table to cell_specimen_ids present in feature_matrix,
    drops duplicates and sets cell_specimen_id as index
    returns dataframe with cells as indices and metadata as columns
    """
    # get metadata for cells in matched cells df
    cell_metadata = cells_table[cells_table.cell_specimen_id.isin(feature_matrix.index.values)]
    cell_metadata = cell_metadata.drop_duplicates(subset='cell_specimen_id')
    cell_metadata = cell_metadata.set_index('cell_specimen_id')
    print(len(cell_metadata), 'cells in cell_metadata for feature_matrix')
    return cell_metadata

def get_cre_lines(cell_metadata):
    """
    get list of cre lines in cell_metadata and sorts them alphabetically
    cell_metadata is a table similar to ophys_cells_table from SDK but can have additional columns based on clustering results
    """
    cre_lines = np.sort(cell_metadata.cre_line.unique())
    return cre_lines

## finding optimal number of clusters

def compute_inertia(a, X, metric='euclidean'):
    W = [np.mean(pairwise_distances(X[a == c, :], metric=metric)) for c in np.unique(a)]
    return np.mean(W)

def compute_gap(clustering, data, k_max=5, n_boots=20, reference_shuffle='all', metric='euclidean'):
    '''
    Computes gap statistic between clustered data (ondata inertia) and null hypothesis (reference intertia).
    input:
        clustering: clustering object that includes "n_clusters" and "fit_predict"
        data: an array of data to be clustered (n samples by n features)
        k_max: (int) maximum number of clusters to test, starts at 1
        n_boots: (int) number of repetitions for computing mean inertias
        reference: (str) what type of shuffle to use, shuffle_dropout_scores,
            None is use random normal distribution
        metric: (str) type of distance to use, default = 'euclidean'
    :return:
        gap_statistics: df of gap statistics metrics
    '''

    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    if isinstance(data, pd.core.frame.DataFrame):
        data_array = data.values
    else:
        data_array = data

    gap_statistics = {}
    reference_inertia = []
    reference_sem = []
    gap_mean = []
    gap_sem = []
    for k in range(1, k_max ):
        local_ref_inertia = []
        for _ in range(n_boots):
            # draw random dist or shuffle for every nboot
            if reference_shuffle is None:
                reference = np.random.rand(*data.shape) * -1
            else:
                reference_df = shuffle_dropout_score(data, shuffle_type=reference_shuffle)
                reference = reference_df.values

            clustering.n_clusters = k
            assignments = clustering.fit_predict(reference)
            local_ref_inertia.append(compute_inertia(assignments, reference, metric=metric))
        reference_inertia.append(np.mean(local_ref_inertia))
        reference_sem.append(sem(local_ref_inertia))

    ondata_inertia = []
    ondata_sem = []
    for k in range(1, k_max ):
        local_ondata_inertia = []
        for _ in range(n_boots):
            clustering.n_clusters = k
            assignments = clustering.fit_predict(data_array)
            local_ondata_inertia.append(compute_inertia(assignments, data_array, metric=metric))
        ondata_inertia.append(np.mean(local_ondata_inertia))
        ondata_sem.append(sem(local_ondata_inertia))

        # compute difference before mean
        gap_mean.append(np.mean(np.subtract(np.log(local_ondata_inertia), np.log(local_ref_inertia))))
        gap_sem.append(sem(np.subtract(np.log(local_ondata_inertia), np.log(local_ref_inertia))))

    # maybe plotting error bars with this metric would be helpful but for now I'll leave it
    gap = np.log(reference_inertia) - np.log(ondata_inertia)

    # we potentially do not need all of this info but saving it to plot it for now
    gap_statistics['gap'] = gap
    gap_statistics['reference_inertia'] = np.log(reference_inertia)
    gap_statistics['ondata_inertia'] = np.log(ondata_inertia)
    gap_statistics['reference_sem'] = reference_sem
    gap_statistics['ondata_sem'] = ondata_sem
    gap_statistics['gap_mean'] = gap_mean
    gap_statistics['gap_sem'] = gap_sem

    return gap_statistics


def get_eigenDecomposition(A, max_n_clusters=25):
    """
    Input:
        A: Affinity matrix from spectral clustering
        max_n_clusters

    :return A tuple containing:
    - the optimal number of clusters by eigengap heuristic
    - all eigen values
    - all eigen vectors

    This method performs the eigen decomposition on a given affinity matrix,
    following the steps recommended in the paper:
    1. Construct the normalized laplacian matrix: L = D−1/2ADˆ −1/2.
    2. Find the eigenvalues and their associated eigen vectors
    3. Identify the maximum gap which corresponds to the number of clusters
    by eigengap heuristic

    References:
    https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    """
    L = csgraph.laplacian(A, normed=True)
    # n_components = A.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Identify the optimal number of clusters as the index corresponding
    # to the larger gap between eigen values
    index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1][:max_n_clusters]
    nb_clusters = index_largest_gap + 1

    return eigenvalues, eigenvectors, nb_clusters


def load_gap_statistic(glm_version, feature_matrix, cre_line='', save_dir=None,
                       metric='euclidean', shuffle_type='all', k_max=25, n_boots=20):
    """
       if gap statistic was computed and file exists in save_dir, load it
       otherwise run spectral clustering n_boots times, for a range of 1 to k_max clusters
       returns dictionary of gap_statistic for each cre line
        shuffle_type is an input to shuffle_dropout_score function, which is used as a null hypothesis or reference data
        metric is distance metric used for in compute gap function
       """
    gap_filename = 'gap_cores_' + cre_line + '_' + metric + '_' + glm_version + '_' + shuffle_type + 'kmax' + str(k_max) + '_' + 'nb' + str(n_boots) + '.pkl'
    gap_path = os.path.join(save_dir, gap_filename)
    if os.path.exists(gap_path):
        print('loading gap statistic scores from', gap_path)
        with open(gap_path, 'rb') as f:
            gap_statistic = pickle.load(f)
            f.close()
        print('done.')
    else:
        gap_statistic = {}
        X = feature_matrix.values
        feature_matrix_shuffled = shuffle_dropout_score(feature_matrix, shuffle_type=shuffle_type)
        reference = feature_matrix_shuffled.values
        # create an instance of Spectral clustering object
        sc = SpectralClustering()
        gap, reference_inertia, ondata_inertia = compute_gap(sc, X, k_max=k_max, reference=reference, metric=metric, n_boots=n_boots)
        gap_statistic['gap'] = gap, 
        gap_statistic['reference_inertia'] = reference_inertia,
        gap_statistic['ondata_inertia'] = ondata_inertia,
        if save_dir is None:
            save_dir = filedir
        save_clustering_results(gap_statistic, filename_string=gap_filename, path=save_dir)
    return gap_statistic

def load_eigengap(feature_matrix, version=version, k_max=25, cre_line=None, save_dir=None, ):
    """
        if eigengap values were computed and file exists in save_dir, load it
        otherwise run get_eigenDecomposition for a range of 1 to k_max clusters
        returns dictionary of eigengap for each cre line = [nb_clusters, eigenvalues, eigenvectors]
        # this doesnt actually take too long, so might not be a huge need to save files besides records
           """

    
    eigengap_filename = 'eigengap_' + version + '_' + 'kmax' + str(k_max) + '.pkl'
    eigengap_path = os.path.join(save_dir, eigengap_filename)
    if os.path.exists(eigengap_path):
        print('loading eigengap values scores from', eigengap_path)
        with open(eigengap_path, 'rb') as f:
            eigengap = pickle.load(f)
            f.close()
        print('done.')
    else:
        eigengap = {} #dictionary
        X = feature_matrix.values
        sc = SpectralClustering(2)  # N of clusters does not impact affinity matrix
        # but you can obtain affinity matrix only after fitting, thus some N of clusters must be provided.
        sc.fit(X)
        A = sc.affinity_matrix_
        eigenvalues, eigenvectors, nb_clusters = get_eigenDecomposition(A, max_n_clusters=k_max)
        eigengap['nb_clusters'] = nb_clusters, 
        eigengap['eigenvalues'] = eigenvalues,
        eigengap['eigenvectors'] = eigenvectors,

        if save_dir is None:
            save_dir = filedir

        save_clustering_results(eigengap, filename_string=eigengap_filename, path=save_dir)

    return eigengap