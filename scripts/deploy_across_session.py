import os
import argparse
import sys
import time
import pandas as pd
import numpy as  np

from simple_slurm import Slurm
import visual_behavior_glm.database as db
import visual_behavior_glm.GLM_across_session as gas

parser = argparse.ArgumentParser(description='deploy glm fits to cluster')
parser.add_argument('--env-path', type=str, default='/home/iryna.yavorska/anaconda3/envs/mfish_glm/', metavar='path to conda environment to use')

def already_fit(cell_id,glm_version):
    filepath = "//allen/programs/braintv/workgroups/nc-ophys/omFish_glm/ophys_glm/v_"+glm_version+"/across_session/"+str(cell_id)+".csv"
    return os.path.exists(filepath) 

if __name__ == "__main__":
    args = parser.parse_args()
    python_executable = "{}/bin/python".format(args.env_path)
    print('python executable = {}'.format(python_executable))
    python_file = "//home/iryna.yavorska/code/visual_behavior_glm_mFish/scripts/across_session.py"
    glm_version = '02_nonrigit_events_sac'
    stdout_basedir = "//allen/programs/braintv/workgroups/nc-ophys/omFish_glm/ophys_glm"
    stdout_location = os.path.join(stdout_basedir, 'job_records_across_session')
    if not os.path.exists(stdout_location):
        print('making folder {}'.format(stdout_location))
        os.mkdir(stdout_location)
    print('stdout files will be at {}'.format(stdout_location))
    cell_table = gas.get_cell_list(glm_version) # sessions that meet FNN+ matched criteria
    cell_ids = cell_table['cell_specimen_id'].unique()

    job_count = 0

    job_string = "--cell {} --version {}"

    n_cell_ids = len(cell_ids)
    print('n_cell_ids = {}'.format(n_cell_ids))
    for cell_id in cell_ids:
        if already_fit(cell_id,glm_version):
            print('already fit, skipping')
        else:
            job_count += 1
            print('starting cluster job for {}, job count = {}'.format(cell_id, job_count))
            job_title = 'cell_{}'.format(cell_id)
            walltime = '2:00:00'
            mem = '100gb'
            job_id = Slurm.JOB_ARRAY_ID
            job_array_id = Slurm.JOB_ARRAY_MASTER_ID
            output = stdout_location+"/"+str(job_array_id)+"_"+str(job_id)+"_"+str(cell_id)+".out"
        
            # instantiate a SLURM object
            slurm = Slurm(
                cpus_per_task=4,
                job_name=job_title,
                time=walltime,
                mem=mem,
                output= output,
                partition="braintv"
            )
    
            args_string = job_string.format(cell_id,glm_version)
            slurm.sbatch('{} {} {}'.format(
                    python_executable,
                    python_file,
                    args_string,
                )
            )
            time.sleep(0.001)
