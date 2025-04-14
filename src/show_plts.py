import optuna
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go

import os

# file_name = "sllmbo_llm_tpe_deepseek-r1:14b_gas_drift_lightgbm_llm_init_anotheragent_succesfull"
# file_name = "sllmbo_llm_tpe_deepseek-r1:14b_gas_drift_lightgbm_llm_init_trial3_anotheragent"
# file_name="sllmbo_llm_tpe_deepseek-r1:14b_gas_drift_lightgbm_llm_init_trial1"
# file_name="sllmbo_llm_tpe_deepseek-r1:8b_gas_drift_lightgbm_llm_init_trial"
# file_name="sllmbo_llm_tpe_llama3:8b_gas_drift_lightgbm_llm_init"
# file_name="optuna_gas_drift_lightgbm_random_init"
def get_study(file_name):
    # load file
    file_path = os.path.join("src", "results", "results_bp", f"{file_name}.db")
    study_name = optuna.get_all_study_names(storage=f"sqlite:///{file_path}")[0]
    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///{file_path}",
    )
    return study

optuna_study = get_study("optuna_gas_drift_lightgbm_random_init")
llama3_study = get_study("sllmbo_llm_tpe_llama3:8b_gas_drift_lightgbm_llm_init")
deepseek14b_study = get_study("sllmbo_llm_tpe_deepseek-r1:14b_gas_drift_lightgbm_llm_init_anotheragent_succesfull")
deepseek8b_study = get_study('sllmbo_llm_tpe_deepseek-r1:8b_gas_drift_lightgbm_llm_init')
no_additional_agent = get_study('sllmbo_llm_tpe_deepseek-r1:14b_gas_drift_lightgbm_llm_init_trial1_succesfull_hybrid_no_additional_agent')
# no_additional_agent = get_study('sllmbo_llm_tpe_deepseek-r1:14b_gas_drift_lightgbm_llm_init_3')

study_names=["Optuna", "LLama3", "Deepseek-r1-14b", "Deepseek-r1-8b", "Deepseek-r1-14b-naieve"]
colors=[
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
]
studies = [optuna_study, llama3_study, deepseek14b_study, deepseek8b_study, no_additional_agent]

num_trials=22
fig = go.Figure()
fig1 = go.Figure()
fig2 = go.Figure()
# for study, name in zip(studies, study_names):
for iter in range(len(studies)):
    study=studies[iter]
    name=study_names[iter]
    color=colors[iter] 

    trials = study.get_trials()
    # if name == "Deepseek-r1-14b-naieve":
    #     breakpoint()
    values = []
    best_values = []
    best_value = 0
    time_taken = []
    for i in range(num_trials):
        # datetime_start=datetime.datetime(2025, 4, 13, 1, 15, 34, 328581)
        # datetime_complete=datetime.datetime(2025, 4, 13, 1, 15, 48, 828586)
        time_to_complete = trials[i].datetime_complete - trials[i].datetime_start
        time_to_complete = time_to_complete.seconds+1e-6*time_to_complete.microseconds
        time_taken.append(time_to_complete)
        if trials[i].value > best_value:
            best_value = trials[i].value
        values.append(trials[i].value)
        best_values.append(best_value)
    # add values to plot
    fig.add_trace(
        go.Scatter(
            x=list(range(len(values))),
            y=values,
            mode="lines+markers",
            name=name,
            line=dict(color=color),
            marker=dict(color=color),
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=list(range(len(values))),
            y=best_values,
            mode="lines+markers",
            name=name,
            line=dict(color=color),
            marker=dict(color=color),
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=list(range(len(values))),
            y=values,
            mode="markers",
            name=name,
            line=dict(color=color),
            marker=dict(color=color),
        )
    )
    if name == "LLama3":
        breakpoint()
        continue
    fig2.add_trace(
        go.Scatter(
            x=list(range(len(values))),
            y=time_taken,
            mode="lines",
            name=name,
            line=dict(color=color),
            marker=dict(color=color),
        )
    )
# graph trials in study
# graph first 20 trials
# fig = optuna.visualization.plot_optimization_history(study)

# set figure title:
fig.update_layout(
    title="Optimization History",
    xaxis_title="Trials",
    yaxis_title="Objective Value",
    showlegend=True,
)

fig1.update_layout(
    title="Optimization History (Best Value)",
    xaxis_title="Trials",
    yaxis_title="Objective Value",
    showlegend=True,
)

# update fig to use log scale on y axis
# fig2.update_yaxes(type="log")
fig2.update_layout(
    title="Time To Optimize",
    xaxis_title="Trials",
    yaxis_title="Time (seconds)",
    showlegend=True,
)

fig.show()
fig1.show()
fig2.show()