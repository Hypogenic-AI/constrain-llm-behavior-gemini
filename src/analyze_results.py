import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc

def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    import string, re
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def match(prediction, ground_truths):
    """Check if prediction matches any ground truth (Exact Match logic after normalization)."""
    norm_pred = normalize_text(prediction)
    for truth in ground_truths:
        norm_truth = normalize_text(truth)
        if norm_truth in norm_pred or norm_pred in norm_truth: # Relaxed inclusion
             return True
    return False

def analyze(args):
    with open(args.input_file, 'r') as f:
        data = json.load(f)
        
    # Lists to store metrics
    consistency_scores = []
    is_hallucination = [] # 1 if Wrong or Impossible, 0 if Correct
    
    answerable_total = 0
    answerable_correct = 0
    impossible_total = 0
    
    for item in data:
        score = item['consistency_score']
        greedy = item['greedy_answer']
        is_imp = item['is_impossible']
        golds = item['gold_answers']
        
        consistency_scores.append(score)
        
        if is_imp:
            # Impossible question.
            # Since we forced the model to answer, ANY answer is technically a "hallucination" 
            # or unsupported by context.
            is_hallucination.append(1)
            impossible_total += 1
        else:
            # Answerable question.
            # Check correctness
            if match(greedy, golds):
                is_hallucination.append(0) # Correct
                answerable_correct += 1
            else:
                is_hallucination.append(1) # Wrong
            answerable_total += 1

    # Convert to numpy
    scores = np.array(consistency_scores)
    labels = np.array(is_hallucination)
    
    # 1. AUC-ROC for detecting Hallucinations/Errors
    if len(np.unique(labels)) > 1:
        roc_auc = roc_auc_score(labels, scores)
    else:
        roc_auc = 0.5
        
    print(f"Total Samples: {len(data)}")
    print(f"Answerable: {answerable_total} (Acc: {answerable_correct/answerable_total if answerable_total else 0:.2%})")
    print(f"Impossible: {impossible_total}")
    print(f"Hallucination/Error Rate (Base): {sum(labels)/len(labels):.2%}")
    print(f"ROC-AUC for Uncertainty Score: {roc_auc:.4f}")
    
    # 2. Risk-Coverage Curve
    # Sort by score (ascending: low score = confident)
    # We want to abstain when score is HIGH.
    # So we sort by score ascending. The first k samples are the ones we KEEP (answer).
    
    sorted_indices = np.argsort(scores) # Low to high
    sorted_labels = labels[sorted_indices] # 0=Correct, 1=Error
    
    risks = []
    coverages = []
    
    n = len(labels)
    current_errors = 0
    
    # Iterate from keeping 1 sample to keeping all
    # Actually, efficient way: cumulative sum
    cum_errors = np.cumsum(sorted_labels)
    
    for k in range(1, n + 1):
        # Keeping k samples (lowest uncertainty)
        coverage = k / n
        error_count = cum_errors[k-1]
        risk = error_count / k
        
        coverages.append(coverage)
        risks.append(risk)
        
    # Calculate AURC (Area Under Risk-Coverage)
    # Ideally Risk decreases as Coverage decreases.
    aurc = auc(coverages, risks) # Note: this standard calculation might vary, usually we want low area under Risk vs Coverage?
    # Actually AURC usually defined as Area under the Risk(Coverage) curve.
    
    print(f"AURC: {aurc:.4f}")
    
    # Save Plot
    plt.figure(figsize=(10, 5))
    
    # Subplot 1: Risk-Coverage
    plt.subplot(1, 2, 1)
    plt.plot(coverages, risks, label=f'AURC = {aurc:.3f}')
    plt.xlabel('Coverage')
    plt.ylabel('Risk (Error Rate)')
    plt.title('Risk-Coverage Curve')
    plt.grid(True)
    plt.legend()
    
    # Subplot 2: Histogram of Scores
    plt.subplot(1, 2, 2)
    plt.hist(scores[labels==0], bins=20, alpha=0.5, label='Correct', color='green')
    plt.hist(scores[labels==1], bins=20, alpha=0.5, label='Error/Impossible', color='red')
    plt.xlabel('Consistency Score (Higher=Uncertain)')
    plt.title('Score Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(args.output_plot)
    print(f"Plot saved to {args.output_plot}")
    
    # Save metrics
    metrics = {
        "total_samples": len(data),
        "base_error_rate": float(sum(labels)/len(labels)),
        "roc_auc": float(roc_auc),
        "aurc": float(aurc),
        "answerable_acc": float(answerable_correct/answerable_total) if answerable_total else 0
    }
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="results/experiment_results_100.json")
    parser.add_argument("--output_plot", type=str, default="results/analysis_plot.png")
    args = parser.parse_args()
    analyze(args)