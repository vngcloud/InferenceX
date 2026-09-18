import unittest
from distribution_metrics import metrics


class DistributionMetricsTests(unittest.TestCase):
    def test_uniform(self):
        m=metrics([8]*256,[i//32 for i in range(256)])
        self.assertEqual(m['rank_max_over_mean'],1)
        self.assertEqual(m['assignment_balance_efficiency'],1)
        self.assertAlmostEqual(m['effective_experts'],256)

    def test_same_expert_skew_different_rank_ownership(self):
        c=[1]*8+[0]*248
        packed=metrics(c,[i//32 for i in range(256)])
        spread=metrics(c,[i%8 for i in range(256)])
        self.assertEqual(packed['rank_max_over_mean'],8)
        self.assertEqual(spread['rank_max_over_mean'],1)
        self.assertEqual(packed['hottest_expert_share'],spread['hottest_expert_share'])
        self.assertEqual(packed['empty_expert_fraction'],.96875)

    def test_empty(self):
        self.assertFalse(metrics([0]*256,[i//32 for i in range(256)])['available'])

    def test_invalid(self):
        with self.assertRaises(ValueError):metrics([-1]*256,[i//32 for i in range(256)])
        with self.assertRaises(ValueError):metrics([1]*256,[8]*256)


if __name__=='__main__':unittest.main()
