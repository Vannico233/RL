import unittest

from chart_rl.protocol import parse_action, tool_action
from chart_rl.reward import relaxed_correctness, score_completion


class ProtocolTests(unittest.TestCase):
    def test_answer_with_comparison_in_thought(self):
        result = parse_action("<think>3 < 5, so choose the larger value.</think>\n<answer> 5 </answer>")
        self.assertEqual((result.kind, result.answer), ("answer", "5"))

    def test_tool_roundtrip(self):
        self.assertEqual(parse_action(tool_action([1, 2, 10, 20])).bbox, [1, 2, 10, 20])

    def test_no_ambiguous_or_fallback_answer(self):
        for value in ("The result is 42", "<answer>1</answer><answer>2</answer>",
                      "<answer>1</answer>more text", "<answer> </answer>",
                      tool_action([1, 2, 3, 4]) + "<answer>2</answer>"):
            with self.subTest(value=value):
                self.assertEqual(parse_action(value).kind, "invalid")

    def test_reject_ambiguous_and_nonfinite_json(self):
        for payload in ('{"name":"x","name":"image_zoom_in_tool","arguments":{"bbox_2d":[1,2,3,4]}}',
                        '{"name":"image_zoom_in_tool","arguments":{"bbox_2d":[NaN,2,3,4]}}',
                        '{"name":"image_zoom_in_tool","arguments":{"bbox_2d":[1,2,3,4],"path":"secret"}}'):
            self.assertEqual(parse_action("<tool_call>" + payload + "</tool_call>").kind, "invalid")


class RewardTests(unittest.TestCase):
    def test_relative_tolerance_and_negative_targets(self):
        for target, prediction, expected in [("100", "105", True), ("100", "105.01", False),
                                               ("-100", "-96", True), ("-100", "100", False)]:
            self.assertEqual(relaxed_correctness(target, prediction), expected)

    def test_reference_zero_and_percent_semantics(self):
        self.assertTrue(relaxed_correctness("0", "0"))
        self.assertFalse(relaxed_correctness("0", "0.0"))
        self.assertTrue(relaxed_correctness("50%", "0.5"))
        self.assertFalse(relaxed_correctness("50%", "50"))

    def test_no_unit_guessing_or_nonfinite_answers(self):
        for gold, pred in [("1,000", "1000"), ("$10", "10"), ("10 kg", "10"),
                           ("nan", "nan"), ("inf", "inf"), ("1e309", "1e309")]:
            self.assertFalse(relaxed_correctness(gold, pred))
        self.assertTrue(relaxed_correctness(" Blue ", "blue"))

    def test_reward_only_final_correctness(self):
        self.assertEqual(score_completion(tool_action([1, 2, 3, 4]), "42").reward, 0)
        self.assertEqual(score_completion("The last number is 42", "42").reward, 0)
        result = score_completion("<answer>blue</answer>", "Blue")
        self.assertTrue(result.relaxed)
        self.assertFalse(result.strict)
        self.assertEqual(result.reward, 1)


if __name__ == "__main__":
    unittest.main()
