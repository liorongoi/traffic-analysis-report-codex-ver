from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from road_traffic.view_grouping import propose


class ImageFixture:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.rows = []

    def add(self, image_id, name, image, view_id=None):
        path = self.folder / f'{image_id}.jpg'
        self.assert_write(cv2.imwrite(str(path), image))
        self.rows.append({'id': image_id, 'name': name, 'capture_utc': None,
                          'view_id': view_id, 'view_name': 'Existing' if view_id else None})

    @staticmethod
    def assert_write(ok):
        if not ok:
            raise RuntimeError('Could not create synthetic image fixture')

    def images(self, mission):
        return list(self.rows)

    def image_path(self, mission, image_id):
        return self.folder / f'{image_id}.jpg'


def make_scene(seed):
    rng = np.random.default_rng(seed)
    image = np.full((360, 480), 65, np.uint8)
    cv2.rectangle(image, (0, 200), (479, 359), 112, -1)
    cv2.line(image, (240, 130), (165, 359), 215, 5)
    cv2.line(image, (240, 130), (315, 359), 215, 5)
    for index in range(8):
        x = index * 60
        cv2.rectangle(image, (x + 5, 18), (x + 43, 125), int(rng.integers(35, 225)), 2)
        cv2.line(image, (x, 35), (x + 48, 88), int(rng.integers(30, 240)), 2)
    for _ in range(100):
        x, y = int(rng.integers(4, 476)), int(rng.integers(4, 195))
        radius, value = int(rng.integers(2, 8)), int(rng.integers(20, 240))
        cv2.circle(image, (x, y), radius, value, -1)
    return image


class ViewGroupingTests(unittest.TestCase):
    def test_proposals_group_same_geometry_across_illumination_and_never_assign(self):
        with tempfile.TemporaryDirectory() as temp:
            library = ImageFixture(temp)
            scene = make_scene(42)
            brighter = np.clip(scene.astype(np.float32) * 0.55 + 55, 0, 255).astype(np.uint8)
            other = make_scene(909)
            library.add('a', 'a.jpg', scene, view_id='existing-view')
            library.add('b', 'b.jpg', brighter)
            library.add('c', 'c.jpg', other)

            result = propose(library, 'mission')
            pairs = {frozenset(row['id'] for row in group['images']) for group in result['groups']}
            self.assertIn(frozenset({'a', 'b'}), pairs)
            self.assertTrue(all(not group['assignments_changed'] for group in result['groups']))
            self.assertEqual([row['view_id'] for row in library.rows], ['existing-view', None, None])
            self.assertIn('lighting', result['illumination_handling'])

    def test_empty_mission_returns_no_groups(self):
        with tempfile.TemporaryDirectory() as temp:
            result = propose(ImageFixture(temp), 'mission')
        self.assertEqual(result['groups'], [])
        self.assertEqual(result['analyzed_images'], 0)


if __name__ == '__main__':
    unittest.main()
