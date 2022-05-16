import numpy as np
import pandas as pd
from tqdm import tqdm
import visual_behavior_glm.GLM_fit_tools as gft
import visual_behavior_glm.GLM_params as glm_params
import visual_behavior_glm.GLM_visualization_tools as gvt
import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.utilities as utilities
import matplotlib.pyplot as plt

# TODO,
# Check computation of dropout scores
# Why does gvt.plot_population_averages have so many replace() calls
# how many rows are in across_df?
# shouldn't plot_population_averages give the same value as groupby.mean()
    # maybe I'm filtering cells somewhere

def make_across_run_params(glm_version):
    '''
        Makes a dummy dictionary with the figure directory hard coded
        This is only used as a quick fix for saving figures
    '''
    figdir = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/v_'+glm_version+'/figures/across_session/'
    run_params = {}
    run_params['version'] = glm_version+'_across'
    run_params['figure_dir'] = figdir[:-1]
    return run_params

def plot_across_summary(across_df,across_run_params,savefig=False):
    '''
        Plots the population average dropout scores by experience and cre line,
        for the high level dropouts. Plots two versions, one with statistics 
        computed for the across session scores, the other for the within session 
        scores. 
    '''
    gvt.plot_population_averages(across_df, across_run_params, dropouts_to_show=[
        'all-images_within','omissions_within','behavioral_within','task_within'],
        across_session=True,stats_on_across=True,savefig=savefig)
    gvt.plot_population_averages(across_df, across_run_params, dropouts_to_show=[
        'all-images_within','omissions_within','behavioral_within','task_within'],
        across_session=True,stats_on_across=False,savefig=savefig)

def fraction_same(across_df):
    '''
        Prints a groupby table of the fraction of cells with coding scores
        that are the same between within and across normalization
    '''
    dropouts = ['omissions','all-images','behavioral','task']

    for dropout in dropouts:
        across_df[dropout+'_same'] = across_df[dropout+'_within'] == across_df[dropout+'_across']
    x = across_df.groupby(['cre_line','experience_level'])[['omissions_same','all-images_same','behavioral_same','task_same']].mean()
    print(x)
    return across_df

def scatter_df(across_df,cell_type, across_run_params,savefig=False):
    '''
        Plots a scatter plot comparing within and across coding scores
        for each of the high level dropouts for cells of <cell_type>

    '''   
 
    across_df = across_df.query('cell_type == @cell_type')

    fig, ax = plt.subplots(2,2,figsize=(11,8))
    plot_dropout(across_df, 'omissions', ax[0,0])
    plot_dropout(across_df, 'all-images', ax[0,1])
    plot_dropout(across_df, 'behavioral', ax[1,0])
    plot_dropout(across_df, 'task', ax[1,1])
    fig.suptitle(cell_type, fontsize=20)

    plt.tight_layout()
    if savefig:
        plt.savefig(across_run_params['figure_dir']+cell_type.replace(' ','_')+'_scatter.png')

def plot_dropout(across_df, dropout, ax):
    ''' 
        Helper function for scatter_df
    '''
    experience_levels = across_df['experience_level'].unique()
    colors = gvt.project_colors()
    for elevel in experience_levels:
        eacross_df = across_df.query('experience_level == @elevel')
        ax.plot(-eacross_df[dropout+'_within'],-eacross_df[dropout+'_across'],'o',color=colors[elevel])
    ax.set_xlabel(dropout+' within',fontsize=18)
    ax.set_ylabel(dropout+' across',fontsize=18)
    ax.tick_params(axis='both',labelsize=16)


def load_cells(glm_version):
    '''
        Loads all cells that have across session coding scores computed.
        prints the cell_specimen_id for any cell that cannot be loaded.

        ARGS
        glm_version (str), name of glm version to use  
    
        RETURNS  
        df  - a dataframe containing the across and within session normalization
        fail_df - a dataframe containing cell_specimen_ids that could not be loaded    
    
    '''

    # 3921 unique cells
    print('Loading list of matched cells')
    cells_table = loading.get_cell_table(platform_paper_only=True).reset_index()
    cells_table = cells_table.query('not passive').copy()
    cells_table = utilities.limit_to_last_familiar_second_novel_active(cells_table)
    cells_table = utilities.limit_to_cell_specimen_ids_matched_in_all_experience_levels(cells_table)
    cells = cells_table['cell_specimen_id'].unique()

    dfs = []
    fail_to_load = []
    print('Loading across session normalized dropout scores')
    for cell in tqdm(cells):
        try:
            filename = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/v_'+glm_version+'/across_session/'+str(cell)+'.csv' 
            score_df = pd.read_csv(filename)
            score_df['cell_specimen_id'] = cell
            dfs.append(score_df)
        except:
            fail_to_load.append(cell)
    print(str(len(fail_to_load))+' cells could not be loaded')

    # concatenate into one data frame, and merge in cell table data
    across_df = pd.concat(dfs)
    across_df =  across_df.drop(columns = ['fit_index']).reset_index(drop=True)
    across_df['identifier'] = [str(x)+'_'+str(y) for (x,y) in zip(across_df['ophys_experiment_id'],across_df['cell_specimen_id'])]
    cells_table['identifier'] = [str(x)+'_'+str(y) for (x,y) in zip(cells_table['ophys_experiment_id'],cells_table['cell_specimen_id'])]
    across_df = pd.merge(across_df, cells_table, on='identifier',suffixes=('','_y'),validate='one_to_one')
    
    # Construct dataframe of cells that could not load, for debugging purposes
    fail_df = cells_table.query('cell_specimen_id in @fail_to_load')   
 
    return across_df, fail_df 

def compute_many_cells(cells,glm_version):
    ''' 
        For each cell_specimen_id in cells, compute the across session normalized dropout scores
        using the model in <glm_version>
    ''' 
    for cell in tqdm(cells):
        try:
            data, score_df = across_session_normalization(cell,glm_version)
        except:
            print(str(cell) +' crashed')
 
def across_session_normalization(cell_specimen_id, glm_version):
    '''
        Computes the across session normalization for a cell
        This is very slow because we have to load the design matrices for each object
        Takes about 3 minutes. 
    
    '''
    run_params = glm_params.load_run_json(glm_version)
    data = get_across_session_data(run_params,cell_specimen_id)
    score_df = compute_across_session_dropouts(data, run_params, cell_specimen_id)
    filename = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/v_'+glm_version+'/across_session/'+str(cell_specimen_id)+'.csv'
    score_df.to_csv(filename)

    return data, score_df

def get_across_session_data(run_params, cell_specimen_id):
    '''
        Loads GLM information for each ophys experiment that this cell participated in.
        Very slow, takes about 3 minutes.
    '''

    # Find which experiments this cell was in
    cells_table = loading.get_cell_table(platform_paper_only=True)
    cells_table = cells_table.query('not passive').copy()
    cells_table = cells_table[cells_table['cell_specimen_id'] == cell_specimen_id]
    cells_table = cells_table.query('last_familiar_active or first_novel or second_novel_active')
    oeids = cells_table['ophys_experiment_id']

    # For each experiment, load the session, design matrix, and fit dictionary
    data = {}
    data['ophys_experiment_id'] = oeids
    print('Loading each experiment, will print a bunch of information about the design matrix for each experiment')
    for oeid in oeids: 
        print('Loading oeid: '+str(oeid))
        session, fit, design = gft.load_fit_experiment(oeid, run_params)       
        data[str(oeid)+'_session'] = session
        data[str(oeid)+'_fit'] = fit
        data[str(oeid)+'_design'] = design

    return data


def compute_across_session_dropouts(data, run_params, cell_specimen_id,clean_df = False):
    '''
        Computes the across session dropout scores
        data                a dictionary containing the session object, fit dictionary, 
                            and design matrix for each experiment
        run_params          the paramater dictionary for this version
        cell_speciemn_id    the cell to compute the dropout scores for
        clean_df            (bool) if True, returns only the within and across session dropout scores, 
                            otherwise returns intermediate values for error checking
    
    '''

    # Set up a dataframe to store across session coding scores
    df = pd.DataFrame()
    df['ophys_experiment_id'] =data['ophys_experiment_id']
    score_df = df.set_index('ophys_experiment_id')
    
    # Get list of dropouts to compute across session coding score
    dropouts = ['omissions','all-images','behavioral','task']

    # Iterate across three sessions to get VE of the dropout and full model
    for oeid in score_df.index.values:
        # Get the full comparison values and test values
        results_df = gft.build_dataframe_from_dropouts(data[str(oeid)+'_fit'], run_params)
        score_df['fit_index'] = np.where(data[str(oeid)+'_fit']['fit_trace_arr']['cell_specimen_id'].values == cell_specimen_id)[0][0]

        # Iterate over dropouts
        for dropout in dropouts:
            score_df.at[oeid, dropout] = results_df.loc[cell_specimen_id][dropout+'__avg_cv_adjvar_test']
            score_df.at[oeid,dropout+'_fc'] = results_df.loc[cell_specimen_id][dropout+'__avg_cv_adjvar_test_full_comparison']
            score_df.at[oeid, dropout+'_within'] = results_df.loc[cell_specimen_id][dropout+'__adj_dropout']

            # Get number of timestamps each kernel was active
            dropped_kernels = set(run_params['dropouts'][dropout]['dropped_kernels'])
            design_kernels = set(data[str(oeid)+'_design'].kernel_dict.keys())
            X = data[str(oeid)+'_design'].get_X(kernels=design_kernels.intersection(dropped_kernels))
            score_df.at[oeid, dropout+'_timestamps'] = np.sum(np.sum(np.abs(X.values),axis=1) > 0)

    # Iterate over dropouts and compute coding scores
    clean_columns = []
    for dropout in dropouts:
        clean_columns.append(dropout+'_within')
        clean_columns.append(dropout+'_across')

        # Adjust variance explained based on number of timestamps
        score_df[dropout+'_pt'] = score_df[dropout]/score_df[dropout+'_timestamps']   
        score_df[dropout+'_fc_pt'] = score_df[dropout+'_fc']/score_df[dropout+'_timestamps'] 

        # Determine which session had the highest variance explained    
        score_df[dropout+'_max'] = score_df[dropout+'_fc_pt'].max()

        # calculate across session coding scores
        score_df[dropout+'_across'] = -(score_df[dropout+'_fc_pt'] - score_df[dropout+'_pt'])/(score_df[dropout+'_max'])
        score_df.loc[score_df[dropout+'_across'] > 0,dropout+'_across'] = 0

        # Cleaning step for low VE dropouts
        score_df.loc[score_df[dropout+'_within'] == 0,dropout+'_across'] = 0

    # All done, cleanup
    if clean_df:
        score_df = score_df[clean_columns].copy()
    return score_df
        
def print_df(score_df):
    '''
        Just useful for debugging
    '''
    dropouts = ['omissions','all-images','behavioral','task']
    for d in dropouts:
        print(score_df[[d+'_within',d+'_across']])

def append_kernel_excitation_across(weights_df, across_df):
    '''
        Appends labels about kernel weights from weights_df onto across_df 
        for some kernels, cells are labeled "excited" or "inhibited" if the average weight over 750ms after
        the aligning event was positive (excited), or negative (inhibited)

        Note that the excited/inhibited labels do not depend on within or across session normalization
        since they are based on the weights from the full model. 

        Additionally computes three coding scores for each kernel:
        kernel_across_positive is the across coding score if the kernel was excited, otherwise 0
        kernel_across_negative is the across coding score if the kernel was inhibited, otherwise 0
        kernel_across_signed is kernel_across_positive - kernel_across_negative

        across_df,_ = gas.load_cells()
        across_df = gas.append_kernel_excitation_across(weights_df, across_df) 
    '''   

    # Merge in three kernel metrics from weights_df 
    across_df = pd.merge(
        across_df,
        weights_df[['identifier','omissions_excited','all-images_excited','task_excited']],
        how = 'inner',
        on = 'identifier',
        validate='one_to_one'
        )
 
    # Use kernel metrics to define signed coding scores
    excited_kernels = ['omissions','task','all-images']
    for kernel in excited_kernels:
        across_df[kernel+'_across_positive'] = across_df[kernel+'_across']
        across_df[kernel+'_across_negative'] = across_df[kernel+'_across']
        across_df.loc[across_df[kernel+'_excited'] != True, kernel+'_across_positive'] = 0
        across_df.loc[across_df[kernel+'_excited'] != False,kernel+'_across_negative'] = 0   
        across_df[kernel+'_across_signed'] = across_df[kernel+'_across_positive'] - across_df[kernel+'_across_negative']

    return across_df
