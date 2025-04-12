import optuna
import matplotlib.pyplot as plt

import os

file_name = "sllmbo_llm_tpe_deepseek-r1:14b_gas_drift_lightgbm_llm_init_anotheragent_succesfull"
file_name = "sllmbo_llm_tpe_deepseek-r1:14b_gas_drift_lightgbm_llm_init_trial3_anotheragent"
file_name="sllmbo_llm_tpe_deepseek-r1:14b_gas_drift_lightgbm_llm_init_trial1"

# load file
file_path = os.path.join("src", "results", f"{file_name}.db")
study_name = optuna.get_all_study_names(storage=f"sqlite:///{file_path}")[0]
study = optuna.load_study(
    study_name=study_name,
    storage=f"sqlite:///{file_path}",
)

# graph trials in study
fig = optuna.visualization.plot_optimization_history(study)
fig.show()
breakpoint()