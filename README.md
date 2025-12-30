# Constraining LLM Behavior via Abstention

## Project Overview
This research project investigates how to prevent Large Language Models (LLMs) from hallucinating by enabling them to **abstain** from answering when uncertain. We implement a **Consistency-Based Abstention** mechanism, where the model's uncertainty is estimated by measuring the consistency of multiple stochastically generated responses.

## Key Findings
- **Consistency Signal**: Evaluating on SQuAD 2.0 (N=100) with Llama-3-8B, we found that consistency scores can detect hallucinations with an **ROC-AUC of 0.68**.
- **Accuracy**: The model achieved 97.8% accuracy on answerable questions but struggled with "unanswerable" ones when forced to answer.
- **Abstention**: Consistency thresholding allows trading off coverage for lower risk (error rate).

## Reproducing Results

### 1. Environment Setup
```bash
uv venv
source .venv/bin/activate
uv sync
```

### 2. Run Experiments
```bash
# Run the experiment pipeline (requires OPENROUTER_API_KEY)
uv run python src/experiment_runner.py --num_samples 100 --output_file experiment_results_100.json
```

### 3. Analyze Results
```bash
# Generate metrics and plots
uv run python src/analyze_results.py --input_file results/experiment_results_100.json
```

## File Structure
- `src/experiment_runner.py`: Main script to run inference and data collection.
- `src/analyze_results.py`: Script to calculate metrics (AUC, Risk-Coverage) and generate plots.
- `src/scoring_utils.py`: Utility for calculating token-overlap consistency scores.
- `results/`: Contains JSON output of experiments and analysis plots.
- `datasets/`: Local copy of SQuAD 2.0.

## Report
See [REPORT.md](REPORT.md) for full details on methodology and analysis.