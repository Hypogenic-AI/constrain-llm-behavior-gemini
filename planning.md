# Research Plan: Constraining LLM Behavior via Consistency-Based Abstention

## Research Question
Can Large Language Models (LLMs) effectively reduce hallucination rates by using consistency-based abstention mechanisms to identify and refuse to answer questions they are uncertain about?

## Background and Motivation
LLMs are prone to "hallucinations"—generating confident but incorrect answers. This is critical in high-stakes domains (legal, medical). While humans can say "I don't know," LLMs are trained to generate text. Recent research (SelfCheckGPT, Conformal Abstention) suggests that "consistency" (agreement among multiple sampled outputs) is a strong proxy for correctness. This project aims to validate this hypothesis using state-of-the-art models via API.

## Hypothesis Decomposition
1.  **H1 (Consistency-Accuracy Correlation)**: There is a positive correlation between the consistency of multiple stochastic samples and the correctness of the greedy generation.
2.  **H2 (Abstention Efficacy)**: By thresholding on consistency scores, we can improve the accuracy of the *answered* set (Selective Accuracy) at the cost of coverage.
3.  **H3 (Unanswerable Detection)**: Consistency scores are significantly lower for "unanswerable" questions (from SQuAD 2.0) compared to answerable ones.

## Proposed Methodology

### Approach
We will implement **Consistency-Based Abstention** (similar to SelfCheckGPT). We will use SQuAD 2.0, which contains both answerable and unanswerable questions, providing a perfect testbed for abstention.

### Experimental Steps
1.  **Setup**: Prepare environment and load SQuAD 2.0 dataset (already in `datasets/`).
2.  **Inference**: For each question in the test set (N=100-200):
    *   Generate 1 **Greedy Answer**.
    *   Generate $k=3$ **Stochastic Samples** (Temp=0.7).
3.  **Scoring**: Calculate **Consistency Score** between the greedy answer and stochastic samples using simple Bag-of-Words (Jaccard) or N-gram overlap.
4.  **Thresholding**: Apply varying thresholds to the consistency score. If score < $T$, output "I don't know".
5.  **Evaluation**: Compute Risk-Coverage curves.

### Baselines
1.  **Naive Strategy**: Always answer (Standard LLM behavior).
2.  **Verbalized Confidence**: (If time permits) Prompt model to output confidence 0-1.

### Evaluation Metrics
1.  **Risk (Error Rate)**: % of *answered* questions that are wrong.
2.  **Coverage**: % of questions answered (not abstained).
3.  **AURC (Area Under Risk-Coverage)**: Integrated metric for trade-off.
4.  **Abstention Accuracy**: % of "unanswerable" questions correctly abstained.

### Statistical Analysis Plan
*   **t-test**: Compare mean consistency scores of Correct vs. Incorrect answers.
*   **Risk-Coverage Curve**: Plot Error Rate vs. Coverage.

## Expected Outcomes
*   We expect consistency scores to be lower for hallucinations and unanswerable questions.
*   Abstention should reduce the error rate on the remaining questions.

## Timeline
*   **Phase 1 (Planning)**: Completed.
*   **Phase 2 (Setup)**: 10 min. Install `openai`, `datasets`, etc.
*   **Phase 3 (Implementation)**: 45 min. Adapt `experiment_runner.py` to use OpenRouter API.
*   **Phase 4 (Experiments)**: 45 min. Run on SQuAD 2.0 (N=100).
*   **Phase 5 (Analysis)**: 30 min. Plot curves, calculate metrics.
*   **Phase 6 (Documentation)**: 30 min. Write REPORT.md.

## Potential Challenges
*   **API Cost/Rate Limits**: We will limit N to 100-200 samples to stay within budget.
*   **Latency**: Sequential API calls can be slow. We will use parallelization if possible, or just wait.

## Success Criteria
*   Successfully running the pipeline on real API.
*   Generating a Risk-Coverage plot showing a trade-off (decreasing error as coverage decreases).