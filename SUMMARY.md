# Research Project: Constraining LLM Behavior via Abstention

## Goal
To investigate if Large Language Models (LLMs) can be motivated to abstain from answering ("I don't know") when they are uncertain or likely to hallucinate, thereby increasing safety and reliability.

## Hypothesis
Using a consistency-based check (sampling multiple outputs) allows the model to detect its own hallucinations and abstain effectively.

## Deliverables

### 1. Research & Resources
- **Literature Review**: `literature_review.md` (5 key papers summarized).
- **Resources Catalog**: `resources.md` (Papers, Datasets, Code).
- **Papers**: PDFs stored in `papers/`.
- **Datasets**: SQuAD 2.0, TruthfulQA, NQ Open stored in `datasets/`.

### 2. Experiment Code
- **Main Script**: `experiment_runner.py`
    - Implements a simplified "SelfCheck" mechanism.
    - Uses N-gram overlap between a greedy generation and stochastic samples to calculate an "Inconsistency Score".
    - Abstains if Score > Threshold.
- **Visualization**: `plot_results.py`
    - Plots Risk (Hallucination Rate) vs Coverage (% Answered).
- **Tests**: `tests/test_experiment_runner.py`

### 3. Usage
Use the provided `Makefile` for a smooth workflow:
```bash
make install  # Install dependencies
make test     # Run unit tests
make run      # Run the experiment (GPT-2 on SQuAD 2.0)
make plot     # Plot the Risk-Coverage curve
```

## Status
- **Setup**: Complete.
- **Ready for Execution**: Yes.
