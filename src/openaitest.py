from openai import OpenAI
import os
import json
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_community.llms import VLLMOpenAI
from langchain_ollama import OllamaLLM
import re


client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
model_name="lightgbm"
problem_description="\nWe are working with a tabular classification dataset for gas drift and there are 6 classes.\nthere is only numerical features.\n\n"
metric="f1_weighted"
direction="maximize"
search_space_dict = {
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

with open(
    os.path.abspath("src/configs/prompts/llm_tpe_sys_prompt.txt"), "r"
) as f:
    system_prompt= f.read().format(
        model_name=model_name,
        problem_description=problem_description,
        metric=metric,
        direction=direction,
    )
param_name = 'num_leaves'
param_range = {"type": "int", "low": 20, "high": 3000}
with open(
    os.path.abspath(
        # "src/configs/prompts/llm_tpe_sample_relative_init_prompt.txt"
        "src/configs/prompts/llm_tpe_sample_independent_init_prompt.txt"
    ),
    "r",
) as f:
    human_input = f.read().format(
        param_name=param_name,
        param_range=param_range,
        # search_space_dict=json.dumps(search_space_dict)
    )
systemMessagePrompt = SystemMessagePromptTemplate.from_template(system_prompt)
humanMessagePrompt = HumanMessagePromptTemplate.from_template(f"{human_input}"),
# prompt = ChatPromptTemplate.from_messages(
#     [
#         systemMessagePrompt,
#         humanMessagePrompt
#     ]
# )

# chat_response = client.chat.completions.create(
#     model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
#     messages=[
#         # {"role": "system", "content": "You are a computer program, and only output json files"},
#         {"role":"system", "content": systemMessagePrompt.prompt.template},
#         # {"role": "user", "content": "generate only a json file with keys being a number 1-7, and values being the name of the corresponding day of the week."},
#         {"role": "user", "content": humanMessagePrompt[0].prompt.template},
#     ]
# )
# print("Chat response:", chat_response)
# '''
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template("{human_input}"),
    ]
)
# llm = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# llm = VLLMOpenAI(
#     # model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
#     model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
#     openai_api_key="EMPTY",
#     openai_api_base="http://localhost:8000/v1",
#     max_tokens=4096
#     # api_key="EMPTY",
#     # base_url="http://localhost:8000/v1"
#     )

llm = OllamaLLM(
    # model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    model="deepseek-r1:14b",
    base_url="http://localhost:11434",
    max_tokens=4096
)
memory = ConversationBufferMemory(return_messages=True)

chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

print(f"human input: {human_input}")
raw_response = chain.run(human_input=human_input)
print(raw_response)
json_pattern = r'json\s*({.*?})\s*'
json_pattern = r"\{.*\}"

match = re.search(json_pattern, raw_response, re.DOTALL)
if match: 
    json_str = match.group() # Extract the JSON string try: 
    import json
    dict = json.loads(json_str) 
breakpoint()
# '''