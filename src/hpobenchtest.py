from hpobench.container.benchmarks.nas.tabular_benchmarks import SliceLocalizationBenchmark
from openai import OpenAI
import optuna
from optuna.trial import TrialState
from langchain.memory import ConversationBufferMemory
import ConfigSpace as CS


# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

b = SliceLocalizationBenchmark(rng=1)
config = b.get_configuration_space(seed=1).sample_configuration()
config_space = b.get_configuration_space(seed=1)
result_dict = b.objective_function(configuration=config, fidelity={"budget": 100}, rng=1)

def optuna_gen_config(trial, config_space):
    config_dict = {}
    for key, value in config_space.items():
        if type(key) is CS.CategoricalHyperparameter:
            conf = trial.suggest_categorical(value.name, value.choices)
            config_dict[key] = conf
        elif type(key) is CS.UniformIntegerHyperparameter:
            conf = trial.suggest_int(value.name, value.lower, value.upper)
        elif type(key) is CS.UniformFloatHyperparameter:
            conf = trial.suggest_float
        else:
            conf = trial.suggest_float(value.name, value.lower, value.upper)
            config_dict[key] = conf
    return CS.Configuration(value=config_dict)

def objective(trial):
    config = optuna_gen_config(trial, config_space)
    result_dict = b.objective_function(configuration=config, fidelity={"budget": 100})
    func_val = result_dict['function_value']

    trial.report(func_val, epoch)
    return func_val 

if __name__ == "__main__":
    study = optuna.create_study(study_name="test", direction="minimize", storage="sqlite:///test.db", load_if_exists=True)
    study.optimize(objective, n_trials=500, timeout=1000)