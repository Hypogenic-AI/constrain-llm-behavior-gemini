import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def analyze_results(args):
    with open(args.results_file, 'r') as f:
        data = json.load(f)
        
    # Data extraction
    scores = []
    is_impossible = []
    correct_preds = [] # For non-impossible questions, did it get it right? (Hard to judge with free gen)
    # Actually, for SQuAD 2.0:
    # If impossible, we want it to abstain.
    # If possible, we want it to answer.
    # We use "consistency_score" as the signal for "Abstain". 
    # High Score -> Inconsistent -> Likely Hallucination or Unknown -> Abstain.
    
    # Let's look at the distribution of scores for Impossible vs Possible
    for item in data:
        scores.append(item['consistency_score'])
        is_impossible.append(item['is_impossible'])
        
    scores = np.array(scores)
    is_impossible = np.array(is_impossible)
    
    # 1. Distribution Plot
    plt.figure(figsize=(10, 6))
    plt.hist(scores[is_impossible], alpha=0.5, label='Impossible (Should Abstain)', bins=20, density=True)
    plt.hist(scores[~is_impossible], alpha=0.5, label='Possible (Should Answer)', bins=20, density=True)
    plt.xlabel('Inconsistency Score')
    plt.ylabel('Density')
    plt.title('Distribution of Consistency Scores')
    plt.legend()
    plt.savefig(f"{args.output_dir}/score_distribution.png")
    print(f"Saved distribution plot to {args.output_dir}/score_distribution.png")
    
    # 2. Risk-Coverage Curve
    # Risk = Error Rate on Answered Questions
    # Coverage = % of Questions Answered
    # We vary threshold T. If score > T, we abstain (don't answer).
    # So we answer if score <= T.
    
    thresholds = np.sort(np.unique(scores))
    coverages = []
    risks = []
    
    for t in thresholds:
        # Answer if score <= t
        answered_indices = scores <= t
        n_answered = np.sum(answered_indices)
        if n_answered == 0:
            continue
            
        coverage = n_answered / len(scores)
        
        # Error calculation
        # Error is defined as:
        # 1. Answering an impossible question (False Positive)
        # 2. Answering a possible question INCORRECTLY (False Negative / Hallucination)
        # However, we don't have ground truth accuracy for the *content* of the answer easily without exact match.
        # But for SQuAD 2.0, "Impossible" questions are a proxy for "Unknown".
        # If we just focus on the task "Identify Unanswerable", then:
        # Positive Class = Impossible (Should Abstain).
        # We are using Score to predict Positive.
        # So we can plot ROC for "Detecting Impossible".
        
        # Let's stick to the "Abstention" task definition:
        # Ideally, we answer ONLY Possible questions.
        # So, if we answer an Impossible question, that's an error.
        # What if we answer a Possible question? We assume it's "safe" to answer (ignoring content correctness for now).
        # This is a simplification, but common in "Selective Prediction" on SQuAD 2.0.
        
        errors = is_impossible[answered_indices] # True if impossible (Error), False if possible (Success)
        risk = np.mean(errors)
        
        coverages.append(coverage)
        risks.append(risk)
        
    plt.figure(figsize=(10, 6))
    plt.plot(coverages, risks, marker='.')
    plt.xlabel('Coverage (Fraction of questions answered)')
    plt.ylabel('Risk (Fraction of answered that were Impossible)')
    plt.title('Risk-Coverage Curve')
    plt.grid(True)
    plt.savefig(f"{args.output_dir}/risk_coverage.png")
    print(f"Saved RC curve to {args.output_dir}/risk_coverage.png")
    
    # 3. AUROC for detecting Impossible questions
    # Score is "Inconsistency". We expect Impossible -> High Score.
    auroc = roc_auc_score(is_impossible, scores)
    print(f"AUROC for detecting Unanswerable: {auroc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    analyze_results(args)
