# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "How to constrain LLM's behavior?".

## Papers
Total papers downloaded: 5

| Title | arXiv | Year | File |
|-------|-------|------|------|
| Mitigating LLM Hallucinations via Conformal Abstention | 2405.01563 | 2024 | papers/2405.01563_Conformal_Abstention.pdf |
| Uncertainty-Based Abstention in LLMs Improves Safety | 2404.10960 | 2024 | papers/2404.10960_Uncertainty_Abstention.pdf |
| Know Your Limits: A Survey of Abstention | 2407.18418 | 2024 | papers/2407.18418_Know_Your_Limits_Survey.pdf |
| Don't Hallucinate, Abstain | 2402.00367 | 2024 | papers/2402.00367_Dont_Hallucinate_Abstain.pdf |
| SelfCheckGPT | 2303.08896 | 2023 | papers/2303.08896_SelfCheckGPT.pdf |

## Datasets
Total datasets downloaded: 3

| Name | Source | Type | Location |
|------|--------|------|----------|
| TruthfulQA | HuggingFace | Generation/MC | datasets/truthful_qa/ |
| SQuAD 2.0 | HuggingFace | QA (with unanswerable) | datasets/squad_v2/ |
| Natural Questions Open | HuggingFace | Open QA | datasets/nq_open/ |

## Code Repositories
Total repositories cloned: 3

| Name | Purpose | Location |
|------|---------|----------|
| SelfCheckGPT | Hallucination Detection Baseline | code/selfcheckgpt/ |
| Falcon (Uncertainty Abstention) | Abstention Implementation | code/uncertainty_abstention_falcon/ |
| Abstention Survey | Survey Resources | code/abstention_survey/ |

## Resource Gathering Notes
- **Search Strategy**: Targeted arXiv for "abstention", "hallucination mitigation", and "selective prediction".
- **Selection Criteria**: Prioritized recent (2024) papers with code and theoretical guarantees (Conformal Abstention).
- **Datasets**: Selected standard benchmarks (SQuAD 2.0, TruthfulQA) that explicitly test for unanswerability or truthfulness.

## Recommendations for Experiment Design
1.  **Task**: Question Answering with Abstention.
2.  **Model**: Use an open-weights model (e.g., Llama-3-8B or Mistral-7B) to implement the abstention mechanisms.
3.  **Metric**: Area Under Risk-Coverage (AURC) and Hallucination Rate @ X% Coverage.
