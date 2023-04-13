#!/bin/bash
# Make sure you run conda activate <env> first
# to run this from an environment where the allenSDK is installed

python deploy_glm_fits.py --version 24_dff_all_L2_optimize_by_session --env-path /home/iryna.yavorska/anaconda3/envs/mfish_glm/ --src-path /home/iryna.yavorska/code/GLM/visual_behavior_glm_mFish --job-end-fraction 1 --run_params True
