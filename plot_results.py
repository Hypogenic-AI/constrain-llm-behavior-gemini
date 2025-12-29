import json
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os

def plot_results(results_file, output_image):
    if not os.path.exists(results_file):
        print(f"Error: Results file '{results_file}' not found.")
        print("Please run 'python experiment_runner.py' first.")
        return

    with open(results_file, 'r') as f:
        data = json.load(f)

    # We want to plot Risk vs Coverage by varying the threshold
    # "Risk" here is defined as the error rate on *answered* questions.
    # For SQuAD 2.0:
    # - Error = Answering an Impossible question OR Getting an Answerable question wrong (we only check the first case roughly here)
    # - Coverage = % of questions answered (not abstained)

    scores = [d['consistency_score'] for d in data]
    is_impossible = [d['is_impossible'] for d in data]
    
    # We will simulate varying the threshold from 0.0 to 1.0
    thresholds = np.linspace(0, 1, 100)
    coverages = []
    risks = []
    
    for t in thresholds:
        # If score > t, we abstain.
        # So we answer if score <= t
        answered_indices = [i for i, s in enumerate(scores) if s <= t]
        
        n_answered = len(answered_indices)
        n_total = len(data)
        
        coverage = n_answered / n_total if n_total > 0 else 0
        
        if n_answered == 0:
            risk = 0 # Define risk as 0 if we answer nothing (or handled as undefined)
        else:
            # Calculate Risk: Proportion of Answered questions that were actually Impossible (Hallucinations)
            # (In a real full eval, we'd also check if the answer to an Answerable question was correct)
            # Here, our simplified metric is: Failure = Answering an Impossible Question.
            
            n_hallucinations = sum(1 for i in answered_indices if is_impossible[i])
            risk = n_hallucinations / n_answered
            
        coverages.append(coverage)
        risks.append(risk)
        
    plt.figure(figsize=(10, 6))
    plt.plot(coverages, risks, marker='.', linestyle='-')
    plt.title('Risk-Coverage Curve (Abstention on SQuAD 2.0)')
    plt.xlabel('Coverage (% of questions answered)')
    plt.ylabel('Risk (Hallucination Rate on answered)')
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    
    # Invert X axis? Usually coverage goes 1.0 -> 0.0
    # But plotting normally is fine.
    
    plt.savefig(output_image)
    print(f"Plot saved to {output_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, default="results/experiment_results.json")
    parser.add_argument("--output_image", type=str, default="results/risk_coverage_curve.png")
    args = parser.parse_args()
    
    plot_results(args.results_file, args.output_image)
