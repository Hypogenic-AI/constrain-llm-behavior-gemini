#!/bin/bash
if [ ! -f "experiment_runner.py" ]; then
    echo "Error: experiment_runner.py not found!"
    exit 1
fi

echo "Running Experiment with default settings..."
echo "Model: gpt2, Samples: 50, Threshold: 0.6"
python experiment_runner.py --model_name gpt2 --num_samples 50 --threshold 0.6

if [ -f "results/experiment_results.json" ]; then
    echo "Experiment complete. Results saved to results/experiment_results.json"
else
    echo "Experiment finished but no results file found. Check for errors."
fi