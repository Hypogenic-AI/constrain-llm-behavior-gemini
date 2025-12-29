# Troubleshooting Guide

This guide addresses common issues you might encounter while running the experiment.

## 1. Out of Memory (OOM) Errors

**Symptom**: `RuntimeError: CUDA out of memory.`
**Cause**: The model (`gpt2` or larger) is too big for your GPU VRAM, or the input sequence is too long.
**Solutions**:
- **Use CPU**: Force the script to use CPU (usually automatic if no GPU, but you can unset `CUDA_VISIBLE_DEVICES`).
- **Use a Smaller Model**:
    ```bash
    python experiment_runner.py --model_name distilgpt2
    ```
- **Reduce Batch Size / Samples**:
    - The script processes one example at a time, but generating N samples might be heavy. Reduce `--num_generations` to 1 or 2.

## 2. Dataset Not Found

**Symptom**: `FileNotFoundError: ... datasets/squad_v2`
**Cause**: The dataset download failed or the directory is missing.
**Solutions**:
- **Verify**: Run `python check_data.py`.
- **Re-download**:
    ```bash
    python download_datasets.py
    ```
- **Check Paths**: Ensure you are running the script from the project root.

## 3. Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'transformers'` (or `datasets`, `torch`, etc.)
**Cause**: Dependencies are not installed in your current environment.
**Solutions**:
- **Install**:
    ```bash
    pip install -r requirements.txt
    ```
- **Check Environment**: Ensure you are using the correct virtual environment (`which python`).

## 4. Slow Execution

**Symptom**: The progress bar moves very slowly.
**Cause**: Running a Transformer model on CPU is slow.
**Solutions**:
- **Reduce Scope**: Run fewer samples for testing.
    ```bash
    python experiment_runner.py --num_samples 5
    ```
- **Use GPU**: Ensure `torch.cuda.is_available()` returns `True` (Check `python -c "import torch; print(torch.cuda.is_available())"`).

## 5. High Hallucination Rate (Risk doesn't drop)

**Symptom**: The "Risk-Coverage Curve" is flat or Risk remains high even when abstaining.
**Cause**:
- The model might be *consistently* wrong (Self-Consistency fails if the model is confidently incorrect).
- The `scoring_utils.py` logic (word overlap) might be too simple.
**Solutions**:
- This is a research finding! Document it.
- Try increasing `--num_generations` (e.g., to 5 or 10) to catch rare correct answers.
