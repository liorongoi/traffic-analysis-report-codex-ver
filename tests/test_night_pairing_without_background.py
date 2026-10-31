import unittest

import cv2
import numpy as np

from road_traffic.night import propose


class NightPairingWithoutBackgroundTests(unittest.TestCase):
    def test_compact_close_lights_form_one_uncertain_pair_hypothesis(self):
        rgb = np.full((100, 200, 3), 25, dtype=np.uint8)
        rgb[40:43, 70:73] = 255
        rgb[40:43, 80:83] = 255

        proposals, details = propose(rgb, None, [])

        self.assertEqual(len(proposals), 1)
        proposal = proposals[0]
        self.assertEqual(proposal["kind"], "light")
        self.assertEqual(proposal["label"], "possible paired vehicle lights")
        self.assertEqual(proposal["light_subtype"], "uncertain_light")
        self.assertEqual(proposal["pairing"], "hypothesis")
        self.assertIn("no compatible view background", proposal["reasons"][-1])
        self.assertFalse(details["background_available"])
        self.assertIn("never a vehicle box or count", details["warning"])

    def test_unpaired_single_light_stays_uncertain_without_background(self):
        rgb = np.full((100, 200, 3), 25, dtype=np.uint8)
        rgb[40:43, 70:73] = 255

        proposals, _ = propose(rgb, None, [])

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["label"], "uncertain light")
        self.assertEqual(proposals[0]["light_subtype"], "uncertain_light")
        self.assertEqual(proposals[0]["pairing"], "single")

    def test_pairing_does_not_promote_a_light_to_vehicle(self):
        rgb = np.full((100, 200, 3), 25, dtype=np.uint8)
        rgb[40:43, 70:73] = 255
        rgb[40:43, 80:83] = 255

        proposals, _ = propose(rgb, None, [])

        self.assertTrue(all(item["kind"] == "light" for item in proposals))
        self.assertFalse(any(item.get("suggested_kind") == "vehicle" for item in proposals))


if __name__ == "__main__":
    unittest.main()
