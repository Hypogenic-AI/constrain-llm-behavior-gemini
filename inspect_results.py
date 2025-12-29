import json
import argparse
import os
import random

def inspect_results(results_file, mode='all', n=5):
    if not os.path.exists(results_file):
        print(f"File {results_file} not found.")
        return

    with open(results_file, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} results from {results_file}")

    # Categorize
    # TP: Abstain = True, Impossible = True (Correctly recognized impossibility)
    # TN: Abstain = False, Impossible = False (Correctly answered)
    # FP: Abstain = True, Impossible = False (Refused to answer a valid question)
    # FN: Abstain = False, Impossible = True (Hallucinated an answer)

    tp = [d for d in data if d['abstain'] and d['is_impossible']]
    tn = [d for d in data if not d['abstain'] and not d['is_impossible']]
    fp = [d for d in data if d['abstain'] and not d['is_impossible']]
    fn = [d for d in data if not d['abstain'] and d['is_impossible']]

    categories = {
        'TP (Correct Abstain)': tp,
        'TN (Correct Answer)': tn,
        'FP (False Refusal)': fp,
        'FN (Hallucination)': fn
    }

    print(f"\nStats:")
    for cat, items in categories.items():
        print(f"  {cat}: {len(items)}")

    if mode == 'stats':
        return

    selected_cats = categories.keys() if mode == 'all' else [mode]
    
    for cat in selected_cats:
        if cat not in categories: continue
        
        items = categories[cat]
        if not items: continue

        print(f"\n=== Inspecting: {cat} ===")
        sample = random.sample(items, min(n, len(items)))
        
        for i, item in enumerate(sample):
            print(f"\n[{i+1}] Question: {item['question']}")
            print(f"    Context Snippet: {item.get('context', 'N/A')[:50]}...")
            print(f"    Greedy Answer: {item['generated_answer']}")
            print(f"    Sampled Answers: {item['sampled_answers']}")
            print(f"    Score: {item['consistency_score']:.4f}")
            print(f"    Decision: {'Abstain' if item['abstain'] else 'Answer'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="results/experiment_results.json")
    parser.add_argument("--mode", type=str, default="all", help="all, stats, or specific category like 'FN (Hallucination)'")
    parser.add_argument("--n", type=int, default=3, help="Number of examples to show per category")
    args = parser.parse_args()

    inspect_results(args.file, args.mode, args.n)
