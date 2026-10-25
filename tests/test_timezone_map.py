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


    def test_offline_city_index_has_verified_supported_country_zones(self):
        path=Path(__file__).parents[1]/'web'/'data'/'timezone-cities.json'
        data=json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(data['source'],'GeoNames cities15000')
        self.assertEqual(data['source_date'],'2026-10-25')
        self.assertEqual(data['license'],'CC BY 4.0')
        self.assertEqual(data['source_archive_sha256'],'9c5e0684a0aeaa1b8a62ec08f2edcea94a425626806b8b249c1d9fc9b09168ec')
        self.assertEqual(data['admin1_source_sha256'],'2e4867f5397acb056ea26377601b57af37cb4c81399770225018e9e439e91ad3')
        self.assertEqual(data['country_counts'],{'Israel':106,'Iran':429,'Lebanon':18,'Latvia':15,'Finland':102})
        self.assertEqual(len(data['cities']),670)
        ids=set()
        for city in data['cities']:
            with self.subTest(city=city['name'],country=city['country']):
                self.assertNotIn(city['id'],ids)
                ids.add(city['id'])
                ZoneInfo(city['zone'])
                self.assertTrue(-90<=city['latitude']<=90)
                self.assertTrue(-180<=city['longitude']<=180)
                self.assertGreater(city['population'],0)
                self.assertTrue(city['region'])
        jerusalem=next(city for city in data['cities'] if city['name']=='Jerusalem' and city['country']=='Israel')
        self.assertEqual(jerusalem['zone'],'Asia/Jerusalem')
        self.assertTrue(jerusalem['aliases'])
        local_aliases={'ירושלים':'Israel','תל אביב':'Israel','حيفا':'Israel','تهران':'Iran','مشهد':'Iran','اصفهان':'Iran','بيروت':'Lebanon','طرابلس':'Lebanon','ריגה':'Latvia','הלסינקי':'Finland'}
        for alias,country in local_aliases.items():
            self.assertTrue(any(city['country']==country and alias in city['aliases'] for city in data['cities']),alias)


if __name__=='__main__':
    unittest.main()
