from datasets import load_from_disk
import os

def check_dataset(path):
    print(f"Checking dataset at: {path}")
    if not os.path.exists(path):
        print("FAIL: Directory not found.")
        return

    try:
        dataset = load_from_disk(path)
        print("SUCCESS: Dataset loaded.")
        print(f"Structure: {dataset}")
        
        if 'validation' in dataset:
            print("--- First Example (Validation) ---")
            print(dataset['validation'][0])
        elif 'train' in dataset:
            print("--- First Example (Train) ---")
            print(dataset['train'][0])
            
    except Exception as e:
        print(f"FAIL: Error loading dataset: {e}")

if __name__ == "__main__":
    check_dataset("datasets/squad_v2")
