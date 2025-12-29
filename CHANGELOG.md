# Changelog

All notable changes to the "Constrain LLM Behavior" project will be documented in this file.

## [1.0.0] - 2025-12-28

### Added
- **Experiment Engine**:
    - `experiment_runner.py`: Main script for consistency-based abstention.
    - `scoring_utils.py`: Decoupled scoring logic.
    - `plot_results.py`: Visualization of Risk-Coverage trade-off.
- **Resources**:
    - 5 key research papers in `papers/`.
    - 3 datasets (`squad_v2`, `truthful_qa`, `nq_open`) in `datasets/`.
    - `CITATIONS.bib` for academic referencing.
- **Documentation**:
    - `literature_review.md`: Synthesis of current research.
    - `README.md`, `SUMMARY.md`: Project overviews.
    - `MANIFEST.md`: Detailed file inventory.
    - `results/README.md`: Template for analysis.
- **Automation & Tools**:
    - `Makefile`: Unified command interface.
    - `setup_env.sh`, `run_experiment.sh`, `test_suite.sh`: Shell helpers.
    - `mock_demo.py`: Logic verification script.
    - `check_data.py`: Dataset integrity checker.
    - `.editorconfig`: Code style consistency.

### Fixed
- Refactored `experiment_runner.py` to allow unit testing without loading heavy ML libraries.
- Improved error handling in shell scripts.
