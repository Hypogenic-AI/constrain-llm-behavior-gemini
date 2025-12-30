# Research Report: Constraining LLM Behavior via Consistency-Based Abstention

## 1. Executive Summary
This research investigated whether consistency-based abstention mechanisms can effectively reduce hallucination rates in Large Language Models (LLMs). By analyzing 100 samples from the SQuAD 2.0 dataset using the Llama-3-8B-Instruct model, we found that consistency scores (derived from stochastic sampling) achieved an ROC-AUC of 0.6838 in distinguishing between correct answers and hallucinations/unanswerable questions. Implementing this abstention mechanism allows for a trade-off where the model can maintain high accuracy on answerable questions while rejecting a significant portion of unanswerable ones, though a perfect separation remains challenging.

## 2. Goal
The primary goal was to test the hypothesis that **consistency across multiple stochastic samples is a reliable proxy for model correctness**, particularly in identifying "unanswerable" questions where the model might otherwise hallucinate. This addresses the critical need for safe AI systems that know their limits and can abstain from answering when uncertain.

## 3. Data Construction
### Dataset
- **Source**: SQuAD 2.0 (Stanford Question Answering Dataset).
- **Composition**: Contains both answerable questions (where the answer is in the context) and unanswerable questions (where the context is relevant but does not contain the answer).
- **Subset Used**: 100 samples from the validation set (45 answerable, 55 unanswerable).

### Preprocessing
- No special preprocessing was applied to the text.
- We constructed prompts that **forced** the model to attempt an answer even for unanswerable questions ("do not say 'I don't know' yet"), effectively simulating a "hallucination induction" scenario to test if the consistency signal could catch it.

## 4. Experiment Description
### Methodology
1.  **Model**: `meta-llama/llama-3-8b-instruct` (via OpenRouter API).
2.  **Inference**:
    *   **Greedy Decode**: Temperature = 0.0 (The candidate answer).
    *   **Stochastic Sampling**: 3 samples with Temperature = 0.7.
3.  **Consistency Scoring**:
    *   Calculated the **Inconsistency Score** (0 to 1) based on token overlap (Jaccard Index) between the greedy answer and the 3 stochastic samples.
    *   Higher Score = Higher Inconsistency = Higher Uncertainty.
4.  **Metric**: Area Under the Receiver Operating Characteristic (ROC-AUC) curve for binary classification (Correct vs. Error/Impossible).

### Evaluation Metrics
- **Risk (Error Rate)**: Proportion of incorrect answers among those not abstained.
- **Coverage**: Proportion of questions answered.
- **AURC**: Area Under Risk-Coverage Curve.

## 5. Result Analysis
### Key Findings
1.  **Strong Baseline Performance**: On *answerable* questions, Llama-3-8B is highly accurate (97.78% accuracy), demonstrating strong reading comprehension.
2.  **Consistency Signal Exists**: The inconsistency score successfully discriminates between correct answers and induced hallucinations with an **ROC-AUC of 0.6838**. This confirms the hypothesis that hallucinations tend to be more variable than grounded answers.
3.  **Trade-off Capabilities**: By setting a threshold on the consistency score, we can filter out a portion of the unanswerable questions. However, the separation is not clean; some hallucinations are "confident" (consistent), and some correct answers are phrased differently across samples (inconsistent).

### Quantitative Results
| Metric | Value |
|--------|-------|
| Total Samples | 100 |
| Base Error Rate (No Abstention) | 56.00% |
| ROC-AUC | 0.6838 |
| AURC | 0.3936 |

*Note: The high base error rate is due to the 55 unanswerable questions which we forced the model to answer incorrectly.*

### Limitations
- **Sample Size**: N=100 is small but sufficient for a pilot validation.
- **Prompt Sensitivity**: The specific wording "answer briefly" might affect variance.
- **Consistent Hallucinations**: If the model is biased towards a specific wrong entity (e.g., a famous person mentioned in the text), it might consistently hallucinate that answer, leading to a low inconsistency score (False Negative for the detector).

## 6. Conclusions
The experiments confirm that **consistency is a viable signal for uncertainty estimation** in LLMs. While not a silver bullet (AUC ~0.68), it provides a "free" signal derived purely from the model's own outputs without needing external verifiers or training. For production systems, combining this with a trained verifier (like a Conformal Predictor) would likely yield better results.

## 7. Next Steps
1.  **Scale Up**: Run on the full SQuAD 2.0 validation set.
2.  **Conformal Prediction**: Use the calibration set to find the exact threshold $T$ that guarantees a specific error rate (e.g., < 5%).
3.  **Prompt Engineering**: Test if "Chain-of-Thought" prompting increases the variance of hallucinations, making them easier to detect.
