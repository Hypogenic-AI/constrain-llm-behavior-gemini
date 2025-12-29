import os
import torch
import json
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from scoring_utils import calculate_inconsistency_score

# Note: We use a simplified token-overlap inconsistency metric (implemented in scoring_utils.py)
# This is conceptually similar to SelfCheckGPT-Ngram (n=1) but avoids heavy dependencies.

def load_resources(model_name, dataset_path, device):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model.to(device)
    
    print(f"Loading dataset from {dataset_path}")
    dataset = load_from_disk(dataset_path)
    return model, tokenizer, dataset

def generate_answer(model, tokenizer, prompt, device, do_sample=False, num_return_sequences=1):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=20, 
            do_sample=do_sample,
            temperature=1.0 if do_sample else None,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id
        )
    
    decoded = []
    for i in range(num_return_sequences):
        text = tokenizer.decode(outputs[i][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        decoded.append(text)
        
    return decoded

def run_experiment(args):
    print("Starting experiment with args:", args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model, tokenizer, dataset = load_resources(args.model_name, args.dataset_path, device)
    
    # Filter dataset
    eval_data = dataset['validation'].select(range(args.num_samples))
    
    results = []
    
    for i, example in enumerate(tqdm(eval_data, desc="Evaluating")):
        question = example['question']
        context = example['context']
        is_impossible = example['answers']['answer_start'] == []
        
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        
        # 1. Greedy
        greedy_responses = generate_answer(model, tokenizer, prompt, device, do_sample=False)
        answer_greedy = greedy_responses[0]
        
        # 2. Sampling
        sampled_answers = generate_answer(model, tokenizer, prompt, device, do_sample=True, num_return_sequences=args.num_generations)
        
        # 3. Score
        score = calculate_inconsistency_score(answer_greedy, sampled_answers)
        
        # 4. Abstain Decision
        abstain = score > args.threshold
        final_response = "I don't know" if abstain else answer_greedy
        
        results.append({
            "question": question,
            "is_impossible": is_impossible,
            "generated_answer": answer_greedy,
            "sampled_answers": sampled_answers,
            "consistency_score": score,
            "abstain": abstain,
            "final_response": final_response
        })
        
    # Metrics
    total = len(results)
    abstain_count = sum(1 for r in results if r['abstain'])
    
    correct_abstentions = sum(1 for r in results if r['abstain'] and r['is_impossible'])
    impossible_count = sum(1 for r in results if r['is_impossible'])
    
    false_abstentions = sum(1 for r in results if r['abstain'] and not r['is_impossible'])
    answerable_count = sum(1 for r in results if not r['is_impossible'])
    
    print("-" * 30)
    print("RESULTS Summary")
    print(f"Total Examples: {total}")
    print(f"Threshold: {args.threshold}")
    print(f"Total Abstained: {abstain_count} ({abstain_count/total:.2%})")
    print("-" * 30)
    print(f"Impossible Questions: {impossible_count}")
    print(f"  - Correctly Abstained: {correct_abstentions} ({correct_abstentions/impossible_count if impossible_count else 0:.2%})")
    print("-" * 30)
    print(f"Answerable Questions: {answerable_count}")
    print(f"  - Incorrectly Abstained (False Positive): {false_abstentions} ({false_abstentions/answerable_count if answerable_count else 0:.2%})")
    
    os.makedirs("results", exist_ok=True)
    output_path = f"results/{args.output_file}"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Abstention Experiment")
    parser.add_argument("--model_name", type=str, default="gpt2-medium", help="Model to use")
    parser.add_argument("--dataset_path", type=str, default="datasets/squad_v2", help="Path to dataset")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of examples to evaluate")
    parser.add_argument("--num_generations", type=int, default=3, help="Number of stochastic samples")
    parser.add_argument("--threshold", type=float, default=0.5, help="Abstention threshold (Score > T -> Abstain)")
    parser.add_argument("--output_file", type=str, default="experiment_results.json", help="Output filename")
    
    args = parser.parse_args()
    run_experiment(args)