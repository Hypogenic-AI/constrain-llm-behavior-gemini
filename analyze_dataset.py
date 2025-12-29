from datasets import load_from_disk
import numpy as np

def analyze_squad(path="datasets/squad_v2"):
    print(f"Loading dataset from {path}...")
    try:
        dataset = load_from_disk(path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # SQuAD 2.0 has 'train' and 'validation' splits
    split = 'validation'
    if split not in dataset:
        print(f"Split '{split}' not found. Available: {dataset.keys()}")
        return

    data = dataset[split]
    total = len(data)
    
    print(f"\n--- Analysis of SQuAD 2.0 ({split}) ---")
    print(f"Total Examples: {total}")

    # Count unanswerable
    # In SQuAD 2.0, unanswerable questions have empty 'answers' text or start index
    unanswerable_count = 0
    question_lengths = []
    context_lengths = []

    for example in data:
        is_impossible = len(example['answers']['answer_start']) == 0
        if is_impossible:
            unanswerable_count += 1
        
        question_lengths.append(len(example['question'].split()))
        context_lengths.append(len(example['context'].split()))

    answerable_count = total - unanswerable_count
    
    print(f"\nClass Distribution:")
    print(f"  Answerable:   {answerable_count} ({answerable_count/total:.2%})")
    print(f"  Unanswerable: {unanswerable_count} ({unanswerable_count/total:.2%})")
    print(f"  -> Baseline Accuracy (Always 'Answer'): {answerable_count/total:.2%}")
    print(f"  -> Baseline Accuracy (Always 'Abstain'): {unanswerable_count/total:.2%}")

    print(f"\nLength Statistics (Words):")
    print(f"  Avg Question Length: {np.mean(question_lengths):.2f}")
    print(f"  Avg Context Length:  {np.mean(context_lengths):.2f}")

if __name__ == "__main__":
    analyze_squad()
