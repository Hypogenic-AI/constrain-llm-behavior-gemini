# Project Manifest

This project explores "How to constrain LLM's behavior" using abstention.

## Core Scripts
- **`experiment_runner.py`**: Main experiment script. Loads SQuAD 2.0, generates answers with GPT-2, calculates consistency scores, and abstains if inconsistent.
- **`scoring_utils.py`**: Shared logic for consistency scoring (Pure Python).
- **`plot_results.py`**: Generates a Risk-Coverage curve from the experiment results.
- **`analyze_dataset.py`**: Calculates stats (e.g., % unanswerable) for SQuAD 2.0.
- **`inspect_results.py`**: Qualitative analysis tool to view specific success/failure cases.
- **`check_data.py`**: Utility to verify dataset integrity.
- **`mock_demo.py`**: Standalone script to demonstrate the scoring logic without LLMs.
- **`tests/test_experiment_runner.py`**: Unit tests for the consistency scoring logic.

## Automation
- **`Makefile`**: Main workflow tool.
- **`setup_env.sh`**: Bash script for setup (alternative to Make).
- **`run_experiment.sh`**: Bash script for execution (alternative to Make).
- **`test_suite.sh`**: Bash script for running unit tests.

## Documentation
- **`experiment/README.md`**: Instructions for running and testing the experiment.
- **`literature_review.md`**: Summary of 5 key papers on the topic.
- **`EXPECTED_RESULTS.md`**: Guide to interpreting experimental outcomes.
- **`TROUBLESHOOTING.md`**: Solutions for common errors.
- **`resources.md`**: Catalog of downloaded papers, datasets, and code.
- **`CITATIONS.bib`**: BibTeX citations for the referenced papers.
- **`CHANGELOG.md`**: Record of project changes.
- **`.editorconfig`**: Editor style configuration.
- **`LICENSE`**: MIT License.
- **`papers/README.md`**: Details on downloaded PDFs.
- **`datasets/README.md`**: Details on downloaded datasets.
- **`code/README.md`**: Details on cloned repositories.

## Data
- **`datasets/`**: Contains `squad_v2` (Validation subset used), `truthful_qa`, `nq_open`.
- **`papers/`**: Contains 5 relevant research papers.
- **`code/`**: Contains `selfcheckgpt` (Reference implementation - not required for default experiment).

## Configuration
- **`requirements.txt`**: Python dependencies.
- **`.gitignore`**: Git configuration (custom for datasets).

## Workflow
1.  **Setup**: `bash setup_env.sh` (or `make install`)
2.  **Verify**: `python check_data.py`
3.  **Run**: `bash run_experiment.sh` (or `make run`)
4.  **Visualize**: `python plot_results.py` (or `make plot`)
