# visual_behavior_glm
Fits a kernel regression model to df/f traces during visual behavior. 

# Installing and setting up the package

## Set up an environment

Before installing, it's recommended to set up a new Python environment with Python 3.7:

For example, using Conda:

    conda create -n mFish_glm python=3.8

Then activate the environment:

    conda activate mFish_glm

## Installation

Currently, we are working with a forked version of AllenSDK that lives on Matt's git page:

    git clone 'https://github.com/mattjdavis/AllenSDK.git't
    cd AllenSDK
    pip install -e .

An additional dependency of the package is `brain_observatory_analysis` (BOA)  
Assuming that most users of this package will also be contributing to BOA, it should also be installed in 'editable' mode:

    git clone https://github.com/AllenInstitute/brain_observatory_analysis.git
    cd brain_observatory_analysis
    pip install -e .

Alternatively, the current master branch could be installed with:

    pip install git+https://github.com/AllenInstitute/brain_observatory_analysis.git

Test that the package was installed properly by importing the GLM class from outside of the visual_behavior_glm directory:

    cd ~
    python
    >>> from visual_behavior_glm.glm import GLM

Please report issues at https://github.com/yavorska-iryna/visual_behavior_glm_mFish/issues

# Use

## Defining new kernels/regressors
- Adding a new kernel requires changes in two places.
- Add the kernel parameters to `make_run_json()` as a dictionary with keys `length` and `offset`
- Define the event times in `add_kernel_by_label()`

## Fit the model
- Make the run json using `delete_rebuild_run_json.py` in  `scripts` with `python delete_rebuild_run_json.py --version <version> --label <a descriptive label> --src-path <path_to_source_code>`
- Start the run for a single session with `python scripts/fit_glm.py --oied <ophys_experiment_id> --version <version>`
- Start the run for all sessions at the command line on hpc-login with `python scripts/deploy_fits.py --glm-version <version> --env <name_of_conda_environment>`
- Collect the results across sessions using `retrieve_results(glm_version=<version>)` from `src/GLM_analysis_tools.py`

## Model Iteration System
- `delete_rebuild_run_json.py` saves a copy of current files, as well as a JSON file with run parameters to `../nc-ophys/visual_behavior/ophys_glm/v_<version>/`

