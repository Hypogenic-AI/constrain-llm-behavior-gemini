import os
import json
import argparse
import time
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk, load_dataset
from openai import OpenAI
from scoring_utils import calculate_inconsistency_score
from dotenv import load_dotenv

load_dotenv()

# Initialize Client (OpenRouter)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

def get_response(model, messages, temperature=0.0, max_tokens=100, n=1):
    """
    Get response from API with retries.
    """
    retries = 3
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n
            )
            return [choice.message.content for choice in response.choices]
        except Exception as e:
            print(f"API Error (Attempt {attempt+1}/{retries}): {e}")
            time.sleep(2 * (attempt + 1))
    return [""] * n

def run_experiment(args):
    print(f"Starting experiment with model: {args.model_name}")
    
    # Load Dataset
    # Try loading from local disk first, else download
    try:
        if os.path.exists(args.dataset_path):
            print(f"Loading local dataset from {args.dataset_path}")
            dataset = load_from_disk(args.dataset_path)
        else:
            print("Local dataset not found, loading from HuggingFace...")
            dataset = load_dataset("squad_v2")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Select validation set
    eval_data = dataset['validation']
    if args.num_samples > 0:
        eval_data = eval_data.select(range(args.num_samples))
    
    results = []
    
    print(f"Evaluating {len(eval_data)} samples...")
    
    for i, example in enumerate(tqdm(eval_data)):
        question = example['question']
        context = example['context']
        is_impossible = len(example['answers']['answer_start']) == 0
        
        # Construct Prompt
        # We ask the model to answer the question based on the context.
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Read the context and answer the question. If the question cannot be answered from the context, answer very briefly with your best guess or what you think is true, but do not say 'I don't know' yet."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"}
        ]
        
        # 1. Greedy Generation (Temp=0)
        greedy_responses = get_response(args.model_name, messages, temperature=0.0, n=1)
        greedy_ans = greedy_responses[0]
        
        # 2. Stochastic Sampling (Temp=0.7)
        # We need N samples. Some APIs don't support n>1 efficiently or at all on some models.
        # We will loop if necessary, but OpenRouter usually supports n.
        # To be safe and generic with OpenRouter models (some ignore n), we loop.
        sampled_ans = []
        for _ in range(args.num_generations):
             resp = get_response(args.model_name, messages, temperature=0.7, n=1)
             sampled_ans.append(resp[0])
             
        # 3. Calculate Consistency Score
        score = calculate_inconsistency_score(greedy_ans, sampled_ans)
        
        # 4. Abstain Decision (Post-hoc)
        # We don't decide here, we save the score to analyze trade-offs later.
        
        results.append({
            "id": example['id'],
            "question": question,
            "is_impossible": is_impossible,
            "greedy_answer": greedy_ans,
            "sampled_answers": sampled_ans,
            "consistency_score": score,
            "gold_answers": example['answers']['text'] if not is_impossible else []
        })
        
        # Save intermediate
        if (i + 1) % 10 == 0:
            with open(f"results/{args.output_file}", "w") as f:
                json.dump(results, f, indent=2)

    # Final Save
    os.makedirs("results", exist_ok=True)
    output_path = f"results/{args.output_file}"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/llama-3-8b-instruct", help="OpenRouter model ID")
    parser.add_argument("--dataset_path", type=str, default="datasets/squad_v2", help="Path to local dataset")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to run")
    parser.add_argument("--num_generations", type=int, default=3, help="Number of samples for consistency")
    parser.add_argument("--output_file", type=str, default="experiment_results.json")
    
    args = parser.parse_args()
    run_experiment(args)
