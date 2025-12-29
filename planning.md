# Research Plan: Constraining LLM Behavior via Consistency-Based Abstention

## 1. Research Question
Can consistency-based abstention mechanisms effectively enable Large Language Models (LLMs) to identify their own knowledge boundaries and abstain from answering unanswerable questions, thereby reducing hallucinations?

## 2. Background and Motivation
LLMs are prone to hallucinations, confidently answering questions they don't know the answer to. For safe deployment, models must be able to say "I don't know". Prior work (SelfCheckGPT, Conformal Abstention) suggests that "consistency" (agreement among multiple sampled outputs) is a strong signal for correctness. If a model hallucinates, its answers across samples tend to diverge. If it knows the answer, they converge. We aim to validate this hypothesis on SQuAD 2.0 (which contains explicit unanswerable questions).

## 3. Hypothesis Decomposition
- **H1**: Correct answers have higher self-consistency scores than hallucinations.
- **H2**: Setting a threshold on consistency score allows the model to selectively abstain, improving the accuracy of the *answered* questions.
- **H3**: There is a trade-off between coverage (answering more questions) and accuracy, which can be tuned via the threshold.

## 4. Proposed Methodology

### Approach
We will use **consistency-based abstention**.
1.  Given a question $x$.
2.  Generate a greedy answer $y_{greedy}$.
3.  Generate $k$ stochastic samples $y_1, ..., y_k$.
4.  Calculate a **consistency score** $S(y_{greedy}, \{y_i\})$ measuring how similar the samples are to the greedy answer.
5.  If $S < T$ (threshold), abstain.

### Experimental Steps
1.  **Setup**: Prepare SQuAD 2.0 dataset (already downloaded).
2.  **Baseline**: Standard Greedy Decoding (always answer).
3.  **Experiment**: Run consistency checks with $k=3$ (or more if feasible).
4.  **Analysis**:
    - Compute distributions of consistency scores for "Answerable" vs "Unanswerable" questions.
    - Plot Risk-Coverage curve.
    - Calculate Accuracy on Answered set at different thresholds.

### Baselines
- **Greedy Decoding**: The standard behavior (coverage = 100%, risk = baseline error rate).
- **Random Abstention** (Theoretical baseline for comparison).

### Evaluation Metrics
- **Abstention Rate**: % of questions declined.
- **Accuracy (Answered)**: Accuracy on the subset of non-abstained questions.
- **AUC-RC (Area Under Risk-Coverage)**: A holistic metric for the trade-off.

### Statistical Analysis Plan
- T-test to compare consistency scores of correct vs. incorrect answers.
- Bootstrap confidence intervals for the AUC-RC.

## 5. Expected Outcomes
- We expect "Unanswerable" questions (from SQuAD 2.0) to have lower consistency scores because the model will hallucinate different random answers or struggle to find a span.
- By filtering low-consistency answers, the accuracy of the remaining answers should increase.

## 6. Timeline
- **Phase 1 (Planning)**: Completed.
- **Phase 2 (Setup)**: Environment and Data verification (10 min).
- **Phase 3 (Implementation)**: Refine `experiment_runner.py` if needed (20 min).
- **Phase 4 (Experiments)**: Run evaluation on SQuAD 2.0 (sample size ~100-200 for speed) (30 min).
- **Phase 5 (Analysis)**: Analyze results and plot curves (20 min).
- **Phase 6 (Reporting)**: Write `REPORT.md` (20 min).

## 7. Potential Challenges
- **Compute**: Generating multiple samples ($k$) multiplies inference time. We will use small sample sizes ($N=100$) and a small model (`gpt2` or `distilgpt2`) to ensure feasibility within the session.
- **Metric**: The "consistency" metric (token overlap) might be too simple. We might need NLI-based consistency if simple overlap fails, but we will start with overlap (n-gram) as implemented.
