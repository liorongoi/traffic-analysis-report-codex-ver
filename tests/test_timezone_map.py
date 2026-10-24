import json
from pathlib import Path
import unittest
from zoneinfo import ZoneInfo


class OfflineTimezoneMapTests(unittest.TestCase):
    def test_bundled_supported_countries_have_valid_geometry_and_iana_zones(self):
        path=Path(__file__).parents[1]/'web'/'data'/'timezone-countries.json'
        data=json.loads(path.read_text(encoding='utf-8'))
        expected={
            'Israel':'Asia/Jerusalem',
            'Iran':'Asia/Tehran',
            'Lebanon':'Asia/Beirut',
            'Latvia':'Europe/Riga',
            'Finland':'Europe/Helsinki',
        }
        self.assertEqual({item['properties']['name']:item['properties']['timezone'] for item in data['features']},expected)
        self.assertEqual(data['source_revision'],'ca3b7150caa5d3cc3fb108a87fc6e3d57bd97d99')
        for feature in data['features']:
            with self.subTest(country=feature['properties']['name']):
                ZoneInfo(feature['properties']['timezone'])
                self.assertIn(feature['geometry']['type'],('Polygon','MultiPolygon'))
                self.assertTrue(feature['geometry']['coordinates'])


if __name__=='__main__':
    unittest.main()
