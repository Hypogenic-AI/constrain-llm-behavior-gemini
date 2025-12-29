import unittest
from scoring_utils import calculate_inconsistency_score

class TestExperimentRunner(unittest.TestCase):
    
    def test_calculate_inconsistency_score_exact_match(self):
        greedy = "Paris is the capital of France"
        samples = ["Paris is the capital of France", "Paris is the capital of France"]
        # Overlap should be 1.0, so Inconsistency should be 0.0
        score = calculate_inconsistency_score(greedy, samples)
        self.assertEqual(score, 0.0)
        
    def test_calculate_inconsistency_score_complete_mismatch(self):
        greedy = "Paris is the capital of France"
        samples = ["London is the capital of UK", "Berlin is in Germany"]
        # Overlap might be small (is, the, of), but definitely not 1.0.
        score = calculate_inconsistency_score(greedy, samples)
        self.assertGreater(score, 0.0)
        
    def test_calculate_inconsistency_score_empty_greedy(self):
        greedy = ""
        samples = ["Something"]
        score = calculate_inconsistency_score(greedy, samples)
        self.assertEqual(score, 1.0)
        
    def test_calculate_inconsistency_score_empty_samples(self):
        greedy = "Answer"
        samples = ["", ""]
        score = calculate_inconsistency_score(greedy, samples)
        self.assertEqual(score, 1.0) # 1 - 0.0 overlap

if __name__ == '__main__':
    unittest.main()
