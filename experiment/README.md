# Experiment: Constraining LLM Behavior via Abstention

This experiment evaluates whether a simple consistency-based mechanism (SelfCheck) can motivate an LLM to abstain ("I don't know") when it is likely to hallucinate.

## Prerequisites

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    # OR
    make install
    ```

2.  **Resources**:
    - Ensure `datasets/squad_v2` is present.
    - Ensure `code/selfcheckgpt` is cloned.

## Workflow (Recommended)

We provide a `Makefile` to streamline the process:

1.  **Install**: `make install`
2.  **Test**: `make test`
3.  **Run**: `make run`
4.  **Visualize**: `make plot`

## Manual Execution

### 1. Running the Experiment

```bash
python experiment_runner.py --model_name gpt2 --num_samples 50 --threshold 0.6
```

**Arguments:**
- `--model_name`: Model identifier (default: `gpt2`)
- `--dataset_path`: Path to dataset (default: `datasets/squad_v2`)
- `--num_samples`: Number of examples to evaluate (default: 20)
- `--num_generations`: Number of stochastic samples for consistency check (default: 3)
- `--threshold`: Inconsistency score threshold (0.0 to 1.0) above which to abstain (default: 0.5)

### 2. Testing

Verify the consistency scoring logic:

```bash
python -m unittest tests/test_experiment_runner.py
```

### 3. Visualization

Generate a Risk-Coverage curve (saved to `results/risk_coverage_curve.png`):

```bash
python plot_results.py
```

## How it Works

1.  **Model**: Generates a greedy answer.
2.  **Sampling**: Generates N stochastic samples.
3.  **Scoring**: Calculates Token IoU between greedy answer and samples.
4.  **Abstention**: If Inconsistency Score > Threshold, output "I don't know".

## Results

Results are saved to `results/experiment_results.json`.
Summary statistics are printed to the console.
