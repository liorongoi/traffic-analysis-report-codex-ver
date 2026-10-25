import json
from pathlib import Path
import unittest
from zoneinfo import ZoneInfo


DATA=Path(__file__).parents[1]/'web'/'data'


class OfflineTimezoneMapTests(unittest.TestCase):
    def test_world_country_geometry_is_complete_and_attributed(self):
        data=json.loads((DATA/'timezone-countries.json').read_text(encoding='utf-8'))
        self.assertEqual(data['feature_count'],239)
        self.assertEqual(len(data['features']),239)
        self.assertEqual(data['source_revision'],'ca3b7150caa5d3cc3fb108a87fc6e3d57bd97d99')
        names={feature['properties']['name'] for feature in data['features']}
        self.assertTrue({'Israel','Iran','Lebanon','Latvia','Finland','United States','Japan','Australia'}<=names)
        for feature in data['features']:
            self.assertIn(feature['geometry']['type'],('Polygon','MultiPolygon'))
            self.assertTrue(feature['geometry']['coordinates'])
            self.assertTrue(feature['properties']['iso_a2'])

    def test_map_click_points_resolve_country_and_timezone_for_target_countries(self):
        countries=json.loads((DATA/'timezone-countries.json').read_text(encoding='utf-8'))['features']
        zones=json.loads((DATA/'timezone-zones.json').read_text(encoding='utf-8'))['features']
        cities=json.loads((DATA/'timezone-cities.json').read_text(encoding='utf-8'))['cities']
        def contains(feature,point):
            geometry=feature['geometry']
            polygons=[geometry['coordinates']] if geometry['type']=='Polygon' else geometry['coordinates']
            def ring_contains(ring):
                inside=False
                for i in range(len(ring)):
                    j=i-1;xi,yi=ring[i];xj,yj=ring[j]
                    if ((yi>point[1])!=(yj>point[1])) and point[0]<(xj-xi)*(point[1]-yi)/(yj-yi)+xi:
                        inside=not inside
                return inside
            return any(ring_contains(poly[0]) and not any(ring_contains(hole) for hole in poly[1:]) for poly in polygons)
        for name,country in [('Tel Aviv','Israel'),('Beirut','Lebanon'),('Tehran','Iran'),('Riga','Latvia'),('Helsinki','Finland'),('Los Angeles','United States')]:
            city=next(item for item in cities if item['name']==name and item['country']==country)
            point=[city['longitude'],city['latitude']]
            self.assertTrue(any(feature['properties']['name']==country and contains(feature,point) for feature in countries),name)
            self.assertTrue(any(feature['properties']['tzid']==city['zone'] and contains(feature,point) for feature in zones),name)

    def test_world_timezone_boundaries_use_valid_iana_ids_and_are_marked_approximate(self):
        data=json.loads((DATA/'timezone-zones.json').read_text(encoding='utf-8'))
        self.assertEqual(data['feature_count'],419)
        self.assertEqual(len(data['features']),419)
        self.assertTrue(data['approximate'])
        self.assertIn('ODbL-1.0',data['license'])
        self.assertEqual(data['source_release'],'2026c')
        zone_ids=set()
        for feature in data['features']:
            zone=feature['properties']['tzid']
            ZoneInfo(zone)
            zone_ids.add(zone)
            self.assertIn(feature['geometry']['type'],('Polygon','MultiPolygon'))
            self.assertTrue(feature['geometry']['coordinates'])
        self.assertIn('Asia/Jerusalem',zone_ids)
        self.assertIn('America/Los_Angeles',zone_ids)

    def test_world_city_index_has_valid_zones_and_localized_aliases(self):
        data=json.loads((DATA/'timezone-cities.json').read_text(encoding='utf-8'))
        self.assertEqual(data['source'],'GeoNames cities15000')
        self.assertEqual(data['source_date'],'2026-10-25')
        self.assertEqual(data['license'],'CC BY 4.0')
        self.assertEqual(data['source_archive_sha256'],'9c5e0684a0aeaa1b8a62ec08f2edcea94a425626806b8b249c1d9fc9b09168ec')
        self.assertEqual(data['admin1_source_sha256'],'2e4867f5397acb056ea26377601b57af37cb4c81399770225018e9e439e91ad3')
        self.assertEqual(data['city_count'],34188)
        self.assertEqual(data['country_count'],244)
        self.assertEqual(len(data['country_names']),252)
        self.assertEqual(len(data['cities']),34188)
        ids=set()
        codes=set()
        for city in data['cities']:
            self.assertNotIn(city['id'],ids)
            ids.add(city['id'])
            codes.add(city['country_code'])
            ZoneInfo(city['zone'])
            self.assertTrue(-90<=city['latitude']<=90)
            self.assertTrue(-180<=city['longitude']<=180)
            self.assertGreaterEqual(city['population'],0)
            self.assertTrue(city['country'])
        self.assertEqual(len(codes),244)
        jerusalem=next(city for city in data['cities'] if city['name']=='Jerusalem' and city['country']=='Israel')
        self.assertEqual(jerusalem['zone'],'Asia/Jerusalem')
        self.assertTrue(jerusalem['aliases'])
        local_aliases={'ירושלים':'Israel','תל אביב':'Israel','حيفا':'Israel','تهران':'Iran','مشهد':'Iran','اصفهان':'Iran','بيروت':'Lebanon','طرابلس':'Lebanon','ריגה':'Latvia','הלסינקי':'Finland'}
        for alias,country in local_aliases.items():
            self.assertTrue(any(city['country']==country and alias in city['aliases'] for city in data['cities']),alias)


if __name__=='__main__':
    unittest.main()
