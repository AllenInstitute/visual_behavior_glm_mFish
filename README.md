# visual_behavior_glm
Fits a kernel regression model to df/f traces during visual behavior. 

## Fit the model
- Make the run json using `src/GLM_fit_tools.make_run_json.py`
- Start the run using `scripts/start_glm.py`
- Collect the results across sessions using `scripts/collect_glm.py`

## Model Iteration System
- `make_run_json(<version>)` saves a copy of current files, as well as a JSON file with run parameters to `../nc-ophys/visual_behavior/ophys_glm/v_<version>/`
   use as: `make_run_json(1, label='Brief description of version #1')
