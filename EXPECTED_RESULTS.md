# Expected Experiment Results

Based on the literature (SelfCheckGPT, Conformal Abstention), here is what you should anticipate when running the experiment.

## 1. Baseline Performance (No Abstention)
- **Coverage**: 100% (The model answers every question).
- **Risk (Error Rate)**: High.
    - SQuAD 2.0 has many unanswerable questions. A standard GPT-2 model (without fine-tuning on SQuAD) will likely hallucinate answers for almost all of them.
    - **Expectation**: Risk $\approx$ % of unanswerable questions in the dataset subset.

## 2. Consistency Scores
- **Answerable Questions**:
    - *Scenario*: "What is the capital of France?"
    - *Greedy*: "Paris"
    - *Samples*: ["Paris", "Paris", "It is Paris"]
    - *Score*: **Low** (close to 0.0). High consistency.
- **Unanswerable/Hallucinated Questions**:
    - *Scenario*: "Who won the 2028 US Election?"
    - *Greedy*: "John Smith"
    - *Samples*: ["Jane Doe", "The Rock", "I don't know"]
    - *Score*: **High** (close to 1.0). High inconsistency.

## 3. Effect of Thresholding (The Risk-Coverage Curve)
As you decrease the threshold (i.e., become stricter, abstaining more easily):

| Threshold | Coverage | Risk (Hallucinations) | Interpretation |
|-----------|----------|-----------------------|----------------|
| **1.0**   | 100%     | High                  | Baseline (Answers everything) |
| **0.8**   | High     | High                  | Only abstains on total gibberish |
| **0.5**   | Medium   | **Decreasing**        | **Sweet Spot**: Abstains on uncertain/hallucinated facts |
| **0.2**   | Low      | Low                   | Too strict: Abstains even on correct answers (False Positives) |
| **0.0**   | 0%       | 0                     | Useless (Abstains on everything) |

## 4. Successful Outcome
The experiment is a success if:
1.  The **Risk-Coverage Curve** slopes upwards (Risk increases as Coverage increases).
2.  You can find a threshold (e.g., 0.5 or 0.6) where:
    - Most **Unanswerable** questions are rejected (True Positives).
    - Most **Answerable** questions are kept (True Negatives).

## Troubleshooting
- **If Risk doesn't drop**: Your model might be "consistently wrong" (hallucinating the same wrong answer every time). This is a known limitation of SelfCheckGPT (see "LLM Hallucinations via Conformal Abstention").
- **If Coverage drops too fast**: The model might be too random/temperature too high, causing even correct answers to vary in phrasing. Adjust `scoring_utils.py` to be more robust (e.g., using NLI or embedding similarity instead of word overlap).
