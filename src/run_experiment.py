from configs.config_dicts import DEFAULT_PARAMS_DICT, ESTIMATORS_DICT, OPTIMIZERS_DICT
from configs.config_enum import (
    DATASET_NAMES,
    LLM_NAMES,
    LLM_TPE_INIT_METHODS,
    ML_MODEL_NAMES,
    OPTIMIZATION_DIRECTIONS,
    OPTIMIZATION_METHODS,
    SLLMBO_METHODS,
)
from configs.config_params import (
    CV_VAR,
    DATASET_VAR,
    DIRECTION_VAR,
    MAX_N_ITERS_WITHOUT_IMPROVEMENT,
    METRIC_FUNC_VAR,
    METRIC_VAR,
    N_SUMMARIZE_ITER,
    NEPTUNE_PROJECT_NAME,
    OPTIMIZATION_CHECK_FUNC_VAR,
    N_TRIALS,
    PROBLEM_DESCRIPTION_VAR,
    PROBLEM_TYPE_VAR,
    RANDOM_STATE,
)
from configs.config_tasks import TASKS_METADATA

from optimizers.optuna_sampler import LLM_TPE_SAMPLER, LLMSampler
from utils import (
    run_single_experiment,
    report_to_neptune,
    visualize_hyperopt_history,
    visualize_llm_history,
)
import optuna
import argparse
import os

def get_param_space(model_name):
    if model_name == ML_MODEL_NAMES.lightgbm.name:
        return {
            "num_leaves": (20, 3000),
            "max_depth": (3, 12),
            "learning_rate": (0.01, 0.3),
            "n_estimators": (100, 1000),
            "min_child_samples": (1, 300),
            "subsample": (0.7, 1.0),
            "colsample_bytree": (0.7, 1.0),
            "reg_alpha": (0, 5),
            "reg_lambda": (0, 5),
        }
    elif model_name == ML_MODEL_NAMES.xgboost.name:
        return {
            "max_depth": (3, 12),
            "learning_rate": (0.01, 0.3),
            "n_estimators": (100, 1000),
            "min_child_weight": (1, 10),
            "subsample": (0.7, 1.0),
            "colsample_bytree": (0.7, 1.0),
            "gamma": (0, 5),
        }
    else:
        raise ValueError(f"Unsupported model: {model_name}")

optimization_method_name=OPTIMIZATION_METHODS.sllmbo.name
# llm_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
llm_name="deepseek-r1:14b"
ml_model_name=ML_MODEL_NAMES.lightgbm.name
dataset_name = DATASET_NAMES.gas_drift.name
problem_metadata = TASKS_METADATA[dataset_name]
X_train, X_test, y_train, y_test = problem_metadata[DATASET_VAR]
cat_cols = X_train.select_dtypes(include=["category"]).columns
problem_description = problem_metadata[PROBLEM_DESCRIPTION_VAR]
metric = problem_metadata[METRIC_VAR]
metric_func = problem_metadata[METRIC_FUNC_VAR]
direction = problem_metadata[DIRECTION_VAR]
optmization_check_func = problem_metadata[OPTIMIZATION_CHECK_FUNC_VAR]
problem_type = problem_metadata[PROBLEM_TYPE_VAR]
cv = problem_metadata[CV_VAR]
default_params = DEFAULT_PARAMS_DICT[ml_model_name]
if len(cat_cols) > 1 and ml_model_name == ML_MODEL_NAMES.xgboost.name:
    default_params["enable_categorical"] = True
estimator = ESTIMATORS_DICT[ml_model_name][problem_type]
optimizer_func = OPTIMIZERS_DICT[optimization_method_name]

llm_tpe_init_method_name = LLM_TPE_INIT_METHODS.llm.name
sllmbo_method_name = SLLMBO_METHODS.sllmbo_llm_tpe.name
patience = None
optimization_args = {
        "X": X_train,
        "y": y_train,
        "model_name": ml_model_name,
        "estimator": estimator,
        "cv": cv,
        "default_params": default_params,
        "n_trials": N_TRIALS,
        "metric": metric,
        "direction": direction,
    }

print(llm_name)
langchain_sampler = LLMSampler(
                llm_name=llm_name,
                model_name=ml_model_name,
                metric=metric,
                direction=direction,
                problem_description=problem_description,
                search_space_dict=get_param_space(ml_model_name),
            )
hybrid_sampler = LLM_TPE_SAMPLER(
    langchain_sampler=langchain_sampler,
    seed=RANDOM_STATE,
    init_method=llm_tpe_init_method_name,
)

experiment_name = f"{sllmbo_method_name}_{llm_name}_{dataset_name}_{ml_model_name}_{llm_tpe_init_method_name}_init"  # noqa: E501
if patience is not None and patience > 0:
    experiment_name += f"_patience_{patience}"
print(experiment_name)
storage = (
    f"sqlite:///{os.path.abspath('src/results')}/{experiment_name}.db"
)

study = optuna.create_study(
    study_name=experiment_name,
    storage=storage,
    direction=direction,
    load_if_exists=False,
    sampler=hybrid_sampler,
)

optimization_args |= {
    "study_name": experiment_name,
    "storage": storage,
    "random_state": RANDOM_STATE,
    "sampler": hybrid_sampler,
    "patience": patience,
}

print(optimizer_func[sllmbo_method_name])
best_params, best_score, best_study, runtime, best_test_score = (
    run_single_experiment(
        optimizer_func=optimizer_func[sllmbo_method_name],
        optimization_args=optimization_args,
        X_test=X_test,
        y_test=y_test,
        metric_func=metric_func,
    )
)

fig_history = optuna.visualization.plot_optimization_history(best_study)
fig_history.update_layout(
    title=f"Optimization History for {experiment_name}"
)