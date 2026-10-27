import unittest

from road_traffic.learning_capabilities import inventory


class LearningCapabilityTests(unittest.TestCase):
    def test_every_shipped_algorithm_declares_each_learning_method(self):
        algorithms = inventory()
        self.assertGreaterEqual(len(algorithms), 8)
        self.assertEqual(len({item['id'] for item in algorithms}), len(algorithms))
        expected = {'fine_tuning', 'full_retraining', 'downstream_classifier_training',
            'background_model_refitting', 'parameter_calibration_only'}
        valid = {'supported', 'not_supported', 'not_implemented'}
        for algorithm in algorithms:
            self.assertTrue(algorithm['name'])
            self.assertTrue(algorithm['role'])
            self.assertEqual(set(algorithm['methods']), expected)
            self.assertEqual(set(algorithm['methods'].values()) - valid, set())
            self.assertTrue(algorithm['available_controls'])
            self.assertTrue(algorithm['limitations'])

    def test_only_reviewed_light_classifier_and_background_builder_claim_learning(self):
        by_id = {item['id']: item for item in inventory()}
        self.assertEqual(by_id['vehicle-detector.yolox-s-coco']['methods']['fine_tuning'], 'not_implemented')
        self.assertEqual(by_id['night-light.multinomial-correction-head']['methods'][
            'downstream_classifier_training'], 'supported')
        self.assertEqual(by_id['night-background.per-view-reference']['methods'][
            'background_model_refitting'], 'supported')
        self.assertIn('excluded', ' '.join(by_id['night-light.multinomial-correction-head']['limitations']))
        plate_limitations = ' '.join(by_id['plate-region.yolox-tiny']['limitations'])
        self.assertIn('unverified suggestion', plate_limitations)
        self.assertIn('Identity lookup is excluded', plate_limitations)


if __name__ == '__main__':
    unittest.main()
