from scoring_utils import calculate_inconsistency_score

def run_mock_demo():
    print("=== Mock Experiment Demo ===")
    print("Demonstrating the Abstention Logic without loading an LLM.\n")

    # Example 1: Consistent / Confident Answer
    print("--- Example 1: Consistent (Confident) ---")
    question_1 = "What is the capital of France?"
    greedy_1 = "Paris"
    samples_1 = ["Paris", "paris", "It is Paris"]
    
    score_1 = calculate_inconsistency_score(greedy_1, samples_1)
    abstain_1 = score_1 > 0.5
    
    print(f"Question: {question_1}")
    print(f"Greedy Answer: {greedy_1}")
    print(f"Sampled Answers: {samples_1}")
    print(f"Inconsistency Score: {score_1:.4f}")
    print(f"Decision: {'ABSTAIN' if abstain_1 else 'ANSWER'}")
    print(f"Final Output: {'I don't know' if abstain_1 else greedy_1}")
    print("\n")

    # Example 2: Inconsistent / Hallucinating Answer
    print("--- Example 2: Inconsistent (Hallucination/Uncertain) ---")
    question_2 = "Who won the 2028 US Election?"
    greedy_2 = "John Smith"
    samples_2 = ["Jane Doe", "The Rock", "I don't know"]
    
    score_2 = calculate_inconsistency_score(greedy_2, samples_2)
    abstain_2 = score_2 > 0.5
    
    print(f"Question: {question_2}")
    print(f"Greedy Answer: {greedy_2}")
    print(f"Sampled Answers: {samples_2}")
    print(f"Inconsistency Score: {score_2:.4f}")
    print(f"Decision: {'ABSTAIN' if abstain_2 else 'ANSWER'}")
    print(f"Final Output: {'I don't know' if abstain_2 else greedy_2}")
    print("\n")

    # Example 3: Empty/Failed Generation
    print("--- Example 3: Empty Generation ---")
    question_3 = "What is the meaning of life?"
    greedy_3 = ""
    samples_3 = ["42", "To be happy"]
    
    score_3 = calculate_inconsistency_score(greedy_3, samples_3)
    abstain_3 = score_3 > 0.5
    
    print(f"Question: {question_3}")
    print(f"Greedy Answer: '{greedy_3}'")
    print(f"Inconsistency Score: {score_3:.4f}")
    print(f"Decision: {'ABSTAIN' if abstain_3 else 'ANSWER'}")
    print("\n")

if __name__ == "__main__":
    run_mock_demo()
