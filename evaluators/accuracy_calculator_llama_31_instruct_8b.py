import argparse
import json
import os
import requests
import torch
from tqdm import tqdm
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

class AccuracyCalculator:

    def __init__(self, model_name, dataset, dataset_path, output_repo, device="cuda:1"): 
        self.model_name = model_name
        self.device = device
        self.output_repo = output_repo
        os.makedirs(self.output_repo, exist_ok=True)
        self.dataset = dataset
        self.dataset_path = dataset_path
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name) 
        # self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        # self.model.eval()

    def calculate_accuracy(self):
        og_accuracy = 0
        rephrased_accuracy = 0
        with open(self.dataset_path, "r") as f:
            data = [json.loads(line) for line in f]
            total = len(data)
            for item in tqdm(data):
                question = item["question"]
                rephrased_question = item["rephrased_question"]
                gold_answer = item["answer"]
                cot = item["cot"]
 
                # prompt_og = f"Give the final answer to the following problem. Answer only with the final number that solves the problem.\n\nProblem: {question}\n\nAnswer:"
                # prompt_rephrased = f"Give the final answer to the following problem. Answer only with the final number that solves the problem.\n\nProblem: {rephrased_question}\n\nAnswer:"

                # prompt_og = f"Solve the following math problem efficiently and clearly:\n\n\
                # - For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n\
                # - For complex problems (3 steps or more):\nUse this step-by-step format:\n\n\
                # ## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n\
                # ## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\n\
                # Regardless of the approach, always conclude with:\n\nTherefore, the final answer is: [answer]\n\n\
                # Where [answer] is just the final number that solves the problem.\n\nProblem: {question}"

                # prompt_rephrased = f"Solve the following math problem efficiently and clearly:\n\n\
                # - For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n\
                # - For complex problems (3 steps or more):\nUse this step-by-step format:\n\n\
                # ## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n\
                # ## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\n\
                # Regardless of the approach, always conclude with:\n\nTherefore, the final answer is: [answer]\n\n\
                # Where [answer] is just the final number that solves the problem.\n\nProblem: {rephrased_question}"

                prompt_og = f"Solve the following math problem efficiently and clearly:\n\n\
                        Use this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n## Step 3: [Concise description]\n[Brief explanation and calculations]\n\n\
                        Only output 3 steps.\n\n\
                        Always conclude with:\n\nTherefore, the final answer is: [answer]\n\n\
                        Where [answer] is just the final number that solves the problem.\n\nProblem: {question}"

                prompt_rephrased = f"Solve the following math problem efficiently and clearly:\n\n\
                        Use this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n## Step 3: [Concise description]\n[Brief explanation and calculations]\n\n\
                        Only output 3 steps.\n\n\
                        Always conclude with:\n\nTherefore, the final answer is: [answer]\n\n\
                        Where [answer] is just the final number that solves the problem.\n\nProblem: {rephrased_question}"
                        
                # Get model answers for both prompts
                model_answer_og , extracted_answer_og = self.get_model_answer(prompt_og)
                model_answer_rephrased , extracted_answer_rephrased = self.get_model_answer(prompt_rephrased)

                if extracted_answer_og is None or model_answer_og is None:
                    is_correct_og = False
                else:
                    is_correct_og = self.compare_answers(extracted_answer_og, gold_answer)
                
                if extracted_answer_rephrased is None or model_answer_rephrased is None:
                    is_correct_rephrased = False
                else:                    
                    is_correct_rephrased = self.compare_answers(extracted_answer_rephrased, gold_answer)


                # Save results                
                result = {
                    "question": question,
                    "rephrased_question": rephrased_question,
                    "gold_answer": gold_answer,
                    "model_answer_og": model_answer_og,
                    "model_answer_rephrased": model_answer_rephrased,
                    "extracted_answer_og": extracted_answer_og,
                    "extracted_answer_rephrased": extracted_answer_rephrased,
                    "is_correct_og": is_correct_og,
                    "is_correct_rephrased": is_correct_rephrased
                }

                with open(os.path.join(self.output_repo, "accuracy_results_with_reasoning.jsonl"), "a") as out_f:
                    out_f.write(json.dumps(result) + "\n")
                og_accuracy += is_correct_og
                rephrased_accuracy += is_correct_rephrased
        if total == 0:
            print("No valid questions to calculate accuracy.")
            return
        og_accuracy /= total
        rephrased_accuracy /= total
        print(f"Original Question Accuracy: {og_accuracy:.4f}")
        print(f"Rephrased Question Accuracy: {rephrased_accuracy:.4f}")

        with open(os.path.join(self.output_repo, "final_accuracy_with_reasoning.json"), "w") as f:
            json.dump({
                "original_question_accuracy": og_accuracy,
                "rephrased_question_accuracy": rephrased_accuracy
            }, f, indent=4)

    # def get_model_answer(self, prompt, max_new_tokens=512):
    #     inputs = self.tokenizer(prompt, add_special_tokens=False,  return_tensors="pt").to(self.device)
 
    #     with torch.no_grad():
    #         outputs = self.model.generate(
    #             **inputs,
    #             max_new_tokens=max_new_tokens,
    #             do_sample=False,
    #             pad_token_id=self.tokenizer.pad_token_id,
    #             eos_token_id=self.tokenizer.eos_token_id,
    #             return_dict_in_generate=True,
    #         )

    #     prompt_length = inputs["input_ids"].shape[1]
    #     generated_tokens = outputs.sequences[0, prompt_length:] 
 
    #     answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    #     extracted_answer = self.extract_final_answer(answer)
    #     return answer, extracted_answer

    def get_model_answer(self, prompt):
        response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "",
        },
        data=json.dumps({
            "model": self.model_name,
            "messages": [
            {
                "role": "user",
                "content": prompt
            }
            ],
        "temperature": 0})
        )
        if "choices" not in response.json():
            print("Error in response:", response.json())
            return None, None
        else:
            answer = response.json()["choices"][0]["message"]["content"]
            return answer, self.extract_final_answer(answer)

    def extract_final_answer(self, model_response):

        answer_pos = model_response.find("Therefore, the final answer is:")

        if answer_pos == -1:
            print("Could not find 'Therefore, the final answer is' in the model response.")
            return None

        final_answer = model_response[answer_pos + len("Therefore, the final answer is:"):].strip()

        response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer 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",
        },
        data=json.dumps({
            "model": "anthropic/claude-sonnet-4",
            "messages": [
            {
                "role": "user",
                "content": "Extract the final answer from the following response. Remove all symbols, commas and text. Only keep the decimal dot if present. Answer only with the final number. \n\nResponse: " + final_answer
            }
            ],
        "temperature": 0})
        )

        if "choices" not in response.json():
            print("Error in response:", response.json())
            return None
        else:
            return response.json()["choices"][0]["message"]["content"]



    def compare_answers(self, model_answer, gold_answer):
        try:
            model_answer = float(model_answer.strip())
            gold_answer = float(gold_answer.strip())
            return model_answer == gold_answer
        except ValueError:
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/llama-3.1-8b-instruct")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--dataset_path", type=str, default="/home/bmg44/RUPBench/rephrased_datasets_llama_31_instruct_405b/gsm8k/rephrased_questions.jsonl")
    parser.add_argument("--output_repo", type=str, default="./accuracy_results_gsm8k_llama_31_instruct_8b")
    args = parser.parse_args()

    accuracy_calculator = AccuracyCalculator(
        model_name=args.model_name,
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        output_repo=args.output_repo
    )
    accuracy_calculator.calculate_accuracy()
