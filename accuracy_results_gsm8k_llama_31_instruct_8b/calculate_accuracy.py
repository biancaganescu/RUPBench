import pandas as pd    
data = pd.read_json(path_or_buf='./accuracy_results_with_reasoning.jsonl', lines=True)


og_count = 0
rephrased_count = 0
for _, item in data.iterrows():
    og_count += item['is_correct_og']
    rephrased_count += item['is_correct_rephrased']

print(f'Original accuracy: {og_count / len(data)}')
print(f'Rephrased accuracy: {rephrased_count / len(data)}')