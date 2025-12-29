# Literature Review: Constraining LLM Behavior via Abstention

## Research Area Overview
The research focuses on motivating Large Language Models (LLMs) to abstain from answering (e.g., saying "I don't know") when they are uncertain or likely to hallucinate. This falls under the domains of **Hallucination Mitigation**, **Selective Prediction**, and **Safety Alignment**. The core hypothesis is that by detecting uncertainty—either through internal signals (logits, entropy) or external consistency checks (sampling multiple outputs)—models can be guided to refuse to answer rather than providing incorrect information.

## Key Papers

### 1. Mitigating LLM Hallucinations via Conformal Abstention
- **arXiv**: 2405.01563
- **Key Contribution**: Proposes a method using **conformal prediction** to calibrate the abstention mechanism. This allows for theoretical guarantees on the maximum hallucination rate (e.g., ensuring hallucinations occur < 5% of the time).
- **Methodology**: Uses self-consistency (similarity of multiple sampled answers) as a non-conformity score. Calibrates a threshold on a validation set to satisfy a user-specified error rate.
- **Results**: Demonstrates valid coverage and reduced hallucination rates on QA tasks compared to heuristic baselines.

### 2. Uncertainty-Based Abstention in LLMs Improves Safety and Reduces Hallucinations
- **arXiv**: 2404.10960
- **Key Contribution**: Distinguishes between **statistical uncertainty** (token-level probability) and **In-Dialogue Uncertainty (InDU)**.
- **Methodology**: Fine-tunes models to output an abstention token or string based on these uncertainty measures.
- **Results**: Shows that InDU is particularly effective for safety-related abstention, while statistical uncertainty helps with factual errors.

### 3. Know Your Limits: A Survey of Abstention in Large Language Models
- **arXiv**: 2407.18418
- **Overview**: A comprehensive survey categorizing abstention approaches into:
    - **Training-based**: RLHF with "I don't know" labels, fine-tuning on unanswerable data.
    - **Inference-time**: Thresholding on confidence scores, self-consistency checks, consistency with retrieved documents.
- **Key Insight**: There is a trade-off between helpfulness (answering everything) and harmlessness/truthfulness (abstaining).

### 4. SelfCheckGPT: Zero-resource black-box hallucination detection
- **arXiv**: 2303.08896
- **Key Contribution**: A widely used baseline that does not require external databases.
- **Methodology**: Samples multiple responses from the LLM (stochastic decoding) and checks if they contradict each other. High contradiction implies hallucination.
- **Relevance**: Can be used as a signal to trigger abstention.

## Common Methodologies

1.  **Self-Consistency / Sampling**: Sampling $k$ responses and measuring their variance or consistency. If variance is high, the model is likely hallucinating -> Abstain.
2.  **Logit-based Confidence**: Using the probability of the generated tokens (avg log-prob) as a confidence score.
3.  **Instruction Tuning / RLHF**: explicitly training the model on (question, "I don't know") pairs.

## Standard Baselines
- **P(True)**: Probability assigned to the generated answer vs. "I don't know".
- **Verbalized Confidence**: Asking the model "Are you sure?" or "Rate your confidence".
- **SelfCheckGPT**: Using consistency as a score.

## Evaluation Metrics
- **Accuracy on Answered**: Accuracy calculated only on questions the model chose to answer.
- **Coverage**: The % of questions the model chose to answer.
- **Risk-Coverage Curve**: Plotting error rate vs. coverage (abstention rate). Area Under Risk-Coverage (AURC) is a key metric.

## Recommended Datasets
- **TruthfulQA**: For testing propensity to mimic common misconceptions vs abstaining.
- **SQuAD 2.0**: Contains explicit "unanswerable" questions based on the context.
- **Natural Questions (NQ)**: Open-domain QA where the model often needs to say "I don't know" if it lacks knowledge (though standard NQ assumes answer exists, the "Open" version is often used for closed-book QA where abstention is valid for obscure facts).

## Gaps and Opportunities
- **Calibration**: Many models are overconfident. Conformal prediction is a promising direction to fix this.
- **Reward Modeling**: Designing a reward function that optimally balances the penalty for a wrong answer vs. the small penalty for abstaining is non-trivial.

## Recommendations for Experiment
1.  **Primary Dataset**: **SQuAD 2.0** (for context-based unanswerability) and **TruthfulQA** (for hallucination/falsehoods).
2.  **Baseline**: **SelfCheckGPT** for detection, and simple **Logit-Probability** thresholding.
3.  **Method to Try**: Implement **Conformal Abstention** (from Paper 1) or a simple **Verifier** that predicts "Answerable/Unanswerable" before generation.
