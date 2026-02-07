import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import json
from pathlib import Path

# put model on GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# initialise model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct").to(device)
model.eval()

# load perturbed dataset from RUPBench
df = pd.read_csv("./perturbed_datasets/GSM8K_RUP.csv")
# sampling 100 for testing, remove for full dataset
df = df.sample(n=100, random_state=42)
question_cols = df.filter(like="question")

# set prompt for cot, adapted from https://huggingface.co/datasets/meta-llama/Llama-3.1-8B-Instruct-evals/viewer/Llama-3.1-8B-Instruct-evals__math__details?row=0

prompt = "Solve the following math problem efficiently and clearly:\n\n\
- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n\
- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\n\
Regardless of the approach, always conclude with:\n\nTherefore, the final answer is: [answer].\n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem:"

out_path = Path("./outputs/rup_llama31_8b_instruct_gsm8k.jsonl")
os.makedirs(out_path.parent, exist_ok=True)

with out_path.open("a", encoding="utf-8") as f:
    for index, row in tqdm(df.iterrows(), total=len(df)):
        for col in question_cols.columns:
            full_prompt = prompt + " " + row[col]
            
            user_msgs = [{"role": "user", "content": full_prompt}]

            user_question = tokenizer.apply_chat_template(
                user_msgs,
                tokenize=False,
                add_generation_prompt=True, # adds the assistant prefix where generation would start
            )

            inputs = tokenizer(user_question, return_tensors="pt", add_special_tokens=False).to(model.device)

            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True)

            prompt_len = inputs["input_ids"].shape[1]
            model_answer = tokenizer.decode(gen.sequences[0, prompt_len:], skip_special_tokens=True)


            record = {
                    "idx": int(index),
                    "variant": col,
                    "question": row[col],
                    # "prompt": full_prompt,
                    "model_answer": model_answer,
                    "gold_answer": row.get("answer") 
                }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()


