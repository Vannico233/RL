import unittest

from PIL import Image

from chart_rl.crop import CropSession


class CropTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (80, 40))
        self.image.putpixel((23, 12), (255, 17, 99))

    def test_mapping_crops_high_resolution_original(self):
        session = CropSession(self.image, (8, 4))
        first = session.execute([2, 1, 6, 3])
        self.assertEqual(first.event["executed_bbox_original"], [20, 10, 60, 30])
        self.assertEqual(first.image.size, (40, 20))
        self.assertEqual(first.image.getpixel((3, 2)), (255, 17, 99))
        second = session.execute([2, 1, 6, 3])
        self.assertEqual(second.image.tobytes(), first.image.tobytes())
        self.assertTrue(second.event["repeated_region"])

    def test_invalid_attempts_exhaust_budget(self):
        session = CropSession(self.image, (8, 4))
        self.assertIsNone(session.execute([-1, 0, 3, 2]).image)
        self.assertIsNone(session.execute([0, 0, 0, 1]).image)
        result = session.execute([0, 0, 4, 4])
        self.assertEqual(result.event["error"], "budget_exhausted")
        self.assertFalse(result.event["consumed_budget"])
        self.assertIsNone(result.image)
        self.assertEqual(session.attempts, 2)

    def test_reject_nonfinite_boolean_and_overflow_without_crash(self):
        for box in ([False, 0, 2, 2], [float("nan"), 0, 2, 2], [0, 0, 10 ** 400, 2],
                    [0, 0, float("inf"), 2], [0, "1", 2, 2], [1, 2]):
            with self.subTest(box=repr(box)[:80]):
                session = CropSession(self.image, (8, 4))
                self.assertIsNone(session.execute(box).image)
                self.assertEqual(session.attempts, 1)

    def test_fractional_bounds_expand_outward(self):
        result = CropSession(self.image, (8, 4)).execute([1.21, 0.31, 4.28, 2.48])
        self.assertEqual(result.event["executed_bbox_original"], [12, 3, 43, 25])

    def test_original_is_owned_by_episode(self):
        session = CropSession(self.image, (8, 4))
        self.image.putpixel((23, 12), (0, 0, 0))
        self.assertEqual(session.execute([2, 1, 6, 3]).image.getpixel((3, 2)), (255, 17, 99))


if __name__ == "__main__":
    unittest.main()
