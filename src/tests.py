import math
import json
import re

def convert_str_list_to_float(str_list):
    clean_str = str_list.strip("[]").replace("'", "").split(',')
    if len(clean_str) > 1:
        digits_before_decimal = int(math.log10(float(clean_str[1]))) + 1 
        value = float(clean_str[0]) + (float(clean_str[1])/(10**digits_before_decimal))
    else:
        value = int(clean_str[0])
    return value

def _extract_key_value_pairs(response: str):
    pairs = {}
    lines = response.split("\n")
    for line in lines:
        match = re.match(r'^{?\s*(["\']?)(\w+)\1\s*:\s*(.+)}?$', line)
        if match:
            key, value = match.group(2), match.group(3)
            value = str(re.findall(r'\d+', value))
            try:
                pairs[key] = json.loads(value)
            except json.JSONDecodeError:
                pairs[key] = value.strip()
    for key, value in pairs.items():
        if isinstance(value, str) and "[" in value:
            try:
                pairs[key] = convert_str_list_to_float(value)
            except ValueError:
                pass
    return pairs

def _parse_llm_response(response: str):
        # breakpoint()
        try:
            # First, try to parse the entire response as JSON
            parsed_response = json.loads(response)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON-like content from the response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                try:
                    parsed_response = json.loads(json_match.group())
                except json.JSONDecodeError:
                    parsed_response = _extract_key_value_pairs(response)
            else:
                parsed_response = _extract_key_value_pairs(response)

        if not isinstance(parsed_response, dict):
            raise ValueError("Response is not a dictionary")

        return parsed_response

def clean_dict_values(input_dict):
    cleaned_dict = {}
    
    for key, value in input_dict.items():
        # If the value is a string of the format "['x', 'y', ...]"
        if isinstance(value, str):
            # Remove square brackets and single quotes, then split by commas
            cleaned_value = value.strip("[]").replace("'", "").split(',')
            
            # Convert each value to a float
            cleaned_dict[key] = [float(v) for v in cleaned_value]
        else:
            cleaned_dict[key] = value
    
    return cleaned_dict

# A = {'num_leaves': "['40']", 'max_depth': "['6']", 'learning_rate': "['0', '05']", 'n_estimators': "['970']", 'min_child_samples': "['30']", 'subsample': "['0', '9195981825434215']", 'colsample_bytree': "['0', '8795975452591109']", 'reg_alpha': "['0', '7800932022121826']", 'reg_lambda': "['0', '7799726016810132']"}

# for key, value in A.items():
#     if isinstance(value, str):
#         A[key] = convert_str_list_to_float(value)

# raw_response= ' <think>```json{"n_estimators": 300}```'
raw_response="""
<think>
Alright, let me try to work through this problem step by step. So, the user is working with a LightGBM model for gas drift classification and wants to maximize the weighted F1 score. They've already provided some trial parameters from Optuna's TPESampler, and they're looking for suggestions on hyperparameter tuning.

First, I need to understand the current state. The best score achieved so far is 0.8135, which isn't too bad but might be improved. The current best parameters include a range of settings across several hyperparameters like num_leaves, max_depth, learning_rate, etc., including n_estimators set at 200.

The task here is to suggest a new value for 'n_estimators' within the specified range of 100 to 1000. The user wants this suggestion in JSON format with just the key and value.

I know that LightGBM's n_estimators controls the number of trees built. More trees can lead to better performance but at the cost of training time. Since the current best is 200, I should consider whether increasing or decreasing this might help.

Looking at some general guidelines, starting with higher learning rates (like 0.1) often requires fewer estimators, but as learning rate decreases, more trees can be beneficial. The user's model has a learning rate of 0.1, which is quite high. So perhaps reducing the number of trees might not be ideal, unless it's leading to overfitting.

Wait, no—actually, with a higher learning rate, each tree contributes more, so sometimes fewer trees can suffice because they learn faster. But in this case, maybe 200 isn't enough, and increasing to around 300-500 might improve performance without overcomplicating things.

Also, considering the problem has 6 classes, it's a multi-class classification. LightGBM handles this by using binary objectives with cross-entropy loss, but sometimes more trees can help in capturing the nuances between classes, especially if they're imbalanced or have complex decision boundaries.

Another thought: perhaps the model hasn't converged yet at 200 estimators. Increasing to 300 might allow it to explore more and improve the F1 score. Alternatively, maybe reducing it could prevent overfitting, but since the current best is already around 81%, I'm not sure if that's the case here.

I should also think about computational resources. The user might be limited by how long each trial takes, so suggesting a value that balances performance and training time is important.

Looking at similar problems and benchmarks, models often find their sweet spot in the n_estimators around 200-500 for LightGBM when using moderate learning rates. So maybe suggesting 300 as the next step would be reasonable.

I should also consider if there's any indication from previous trials about whether increasing or decreasing n_estimators had a positive effect. Since I don't have that data, I'll go with a standard approach and suggest 300 as a midpoint between 200 and 400, which is within the given range.

So, putting it all together, suggesting 'n_estimators' set to 300 seems like a logical next step.
</think>

{"n_estimators": 300}

**Step-by-Step Explanation:**

1. **Understand the Current Model:** The current best parameters include `n_estimators=200`, which has achieved an F1 score of ~0.8135.

2. **Assess n_estimators Impact:** This hyperparameter determines how many trees are built. More trees can improve model performance but may risk overfitting or increased training time.

3. **Consider Learning Rate and Model Complexity:** With a high learning rate (0.1), each tree contributes significantly, so more trees might not always be better. However, the dataset has 6 classes, suggesting complexity that could benefit from more trees.

4. **Trial and Error Approach:** Since Optuna's TPESampler suggests parameters, exploring a higher value like 300 could help improve performance without excessive overfitting risk.

5. **Choose Midpoint for Balance:** Selecting 300 as the next trial balances between the current 200 and a higher number to explore potential improvements in model accuracy.

**Answer:**
```json
{
"n_estimators": 300
}
```
"""
response = _parse_llm_response(raw_response)
if isinstance(response, dict):
    response= list(response.values())[0]
breakpoint()

