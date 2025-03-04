import os
import argparse
import sys
import time
import pandas as pd
import numpy as  np

import visual_behavior_glm.GLM_params as glm_params
import visual_behavior_glm.GLM_fit_tools as gft
from simple_slurm import Slurm
import visual_behavior_glm.database as db
from brain_observatory_analysis.ophys.experiment_loading import start_lamf_analysis
from brain_observatory_analysis.dev import data_selection_tools as dst
# from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache
# from mindscope_qc.data_access import behavior_ophys_experiment_dev as BOE_dev

parser = argparse.ArgumentParser(description='deploy glm fits to cluster')
parser.add_argument('--env-path', type=str, default='/home/iryna.yavorska/anaconda3/envs/mfish_glm/', metavar='path to conda environment to use')
parser.add_argument('--version', type=str, default='02_nonrigit_events_sac', metavar='glm version')
parser.add_argument(
    '--src-path', 
    type=str, 
    default='/home/iryna.yavorska/code/visual_behavior_glm_mFish',
    metavar='src_path',
    help='folder where code lives'
)
parser.add_argument(
    '--force-overwrite', 
    action='store_true',
    default=True,
    dest='force_overwrite', 
    help='Overwrites existing fits for this version if enabled. Otherwise only experiments without existing results are fit'
)
parser.add_argument(
    '--testing', 
    action='store_true',
    default=False,
    dest='testing', 
    help='If this flag is called, only 10 test sessions will be deployed'
)
parser.add_argument(
    '--targeted-restart',
    action='store_true',
    default=False,
    dest='targeted_restart',
    help='Loads a list of ophys_experiment_ids to start'
)
parser.add_argument(
    '--use-previous-fit', 
    action='store_true',
    default=True,
    dest='use_previous_fit', 
    help='use previous fit if it exists (boolean, default = False)'
)
parser.add_argument(
    '--job-start-fraction', 
    type=float, 
    default=0.0,
    metavar='start_fraction',
    help='which fraction of all jobs to start on. useful if splitting jobs amongst users. Default = 0.0'
)
parser.add_argument(
    '--job-end-fraction', 
    type=float, 
    default=1.0,
    metavar='end_fraction',
    help='which fraction of all jobs to end on. useful if splitting jobs amongst users. Default = 1.0'
)
parser.add_argument(
    '--run_params',
    type=bool,
    default=False,
    metavar='run_params',
    help='create run params json file'
)

def calculate_required_mem(roi_count):
    '''calculate required memory in GB'''
    return 12 + 0.25*roi_count

def calculate_required_walltime(roi_count):
    '''calculate required walltime in hours'''
    estimate= 10 + 0.3*roi_count
    return np.min([estimate,48]) 

def select_experiments_for_testing(returns = 'experiment_ids'):
    '''
    This function will return 10 hand-picked experiment IDs to use for testing purposes.
    This will allow multiple versions to test against the same small set of experiments.

    Experiments were chosen as follows:
        2x OPHYS_2_passive
        2x OPHYS_5_passive
        2x active w/ fraction engaged < 0.05 (1 @ 0.00, 1 @ 0.02)
        2x active w/ fraction engaged > 0.99 (1 @ 0.97, 1 @ 0.98)
        2x active w/ fraction engaged in range (0.4, 0.6) (1 @ 0.44, 1 @ 0.59)

    Parameters:
    ----------
    returns : str
        either 'experiment_ids' or 'dataframe'

    Returns:
    --------
    if returns == 'experiment_ids' (default)
        list of 10 pre-chosen experiment IDs
    if returns == 'dataframe':
        experiment table for 10 pre-chosen experiments
    '''

    test_experiments = pd.read_csv('//allen/programs/braintv/workgroups/nc-ophys/omFish_glm/ophys_glm/experiments_for_testing.csv')

    if returns == 'experiment_ids':
        return test_experiments['ophys_experiment_id'].unique()
    elif returns == 'dataframe':
        return test_experiments

def get_roi_count(ophys_experiment_id):
    '''
    a LIMS query to get the valid ROI count for a given experiment
    '''
    query= 'select * from cell_rois where ophys_experiment_id = {}'.format(ophys_experiment_id)
    df = db.lims_query(query)
    return df['valid_roi'].sum()

def already_fit(oeid, version):
    '''
    check the weight_matrix_lookup_table to see if an oeid/glm_version combination has already been fit
    returns a boolean
    '''
    conn = db.Database('omFish_glm')
    coll = conn['ophys_glm']['weight_matrix_lookup_table']
    document_count = coll.count_documents({'ophys_experiment_id':int(oeid), 'glm_version':str(version)})
    conn.close()
    return document_count > 0






if __name__ == "__main__":
    args = parser.parse_args()
    python_executable = "{}/bin/python".format(args.env_path)
    print('python executable = {}'.format(python_executable))
    python_file = "{}/scripts/fit_glm.py".format(args.src_path)

    stdout_basedir = "//allen/programs/braintv/workgroups/nc-ophys/omFish_glm/ophys_glm"
    stdout_location = os.path.join(stdout_basedir, 'job_records_{}'.format(args.version))
    if not os.path.exists(stdout_location):
        print('making folder {}'.format(stdout_location))
        os.mkdir(stdout_location)
    print('stdout files will be at {}'.format(stdout_location))

    if args.run_params:
        from visual_behavior_glm import GLM_params
        GLM_params.make_run_json(VERSION=args.version,src_path=args.src_path)

    if args.testing:
        experiments_table = select_experiments_for_testing(returns = 'dataframe')
    elif args.targeted_restart:
        print('Using targeted restart list')
        restart_table = '//allen/programs/braintv/workgroups/nc-ophys/omFish_glm/ophys_glm/v_'+args.version+'/restart_table.csv'
        print('Using experiments from: '+restart_table)
        experiments_table = pd.read_csv(restart_table)
        print('{} experiments to restart'.format(len(experiments_table)))
    else:
       
        # cache = VisualBehaviorOphysProjectCache.from_lims()
        # experiments_table = cache.get_ophys_experiment_table()
        experiments_table = start_lamf_analysis()
        print('total number of oeids = {}'.format(len(experiments_table)))
        run_params = glm_params.load_run_json(args.version)
        projects = gft.define_project_codes()
        cre_lines = gft.define_cre_lines()
        experience_levels = gft.define_experience_levels()
        experiments_table = experiments_table[(experiments_table.project_code.isin(projects)) &
                                               (experiments_table.experience_level.isin(experience_levels))]
        # experiments_table = experiments_table[(experiments_table.mouse_name.isin(['Copper']))]
        print('after selection number of oeids = {}'.format(len(experiments_table)))
        # if run_params['include_4x2_data']:
        #     print('including 4x2 data')
        #     experiments_table = experiments_table[(experiments_table.reporter_line!="Ai94(TITL-GCaMP6s)")].reset_index()      
        # else:
        #     experiments_table = experiments_table[(experiments_table.project_code!="VisualBehaviorMultiscope4areasx2d")&(experiments_table.reporter_line!="Ai94(TITL-GCaMP6s)")].reset_index()
    print('experiments table loaded')

    # get ROI count for each experiment
    experiments_table.reset_index(inplace=True, drop=False)
    experiments_table['roi_count'] = experiments_table['ophys_experiment_id'].map(lambda oeid: get_roi_count(oeid))
    print('roi counts extracted')

    job_count = 0

    if args.use_previous_fit:
        job_string = "--oeid {} --version {} --use-previous-fit"
    else:
        job_string = "--oeid {} --version {}"

    experiment_ids = experiments_table['ophys_experiment_id'].values
    n_experiment_ids = len(experiment_ids)
    print('refitting copper')
    # experiment_ids = [1193067624, 1193067622, 1193067630, 1193067628, 1193067627, 1193067631, 1193067625, 1193067634]
    for experiment_id in experiment_ids: #[int(n_experiment_ids * args.job_start_fraction): int(n_experiment_ids * args.job_end_fraction)]:

        # calculate resource needs based on ROI count
        roi_count = experiments_table.query('ophys_experiment_id == @experiment_id').iloc[0]['roi_count']

        if args.force_overwrite or not already_fit(experiment_id, args.version):
            job_count += 1
            print('starting cluster job for {}, job count = {}'.format(experiment_id, job_count))
            job_title = 'oeid_{}_fit_glm_v_{}'.format(experiment_id, args.version)
            walltime = '{}:00:00'.format(int(np.ceil((calculate_required_walltime(roi_count)))))
            mem = '{}gb'.format(int(np.ceil((calculate_required_mem(roi_count)))))
            job_id = Slurm.JOB_ARRAY_ID
            job_array_id = Slurm.JOB_ARRAY_MASTER_ID
            output = stdout_location+"/"+str(job_array_id)+"_"+str(job_id)+"_"+str(experiment_id)+".out"
    
            # instantiate a SLURM object
            slurm = Slurm(
                cpus_per_task=16,
                job_name=job_title,
                time=walltime,
                mem=mem,
                output= output,
                partition="braintv"
            )

            args_string = job_string.format(experiment_id, args.version)
            slurm.sbatch('{} {} {}'.format(
                    python_executable,
                    python_file,
                    args_string,
                )
            )
            time.sleep(0.1)
