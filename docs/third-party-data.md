# Bundled geographic data

The offline mission wizard uses three local datasets. It makes no tile, geocoding, or timezone API requests. The timezone boundary layer is an approximate visualization; operators can edit the selected country and IANA zone, and the polygons are not authoritative legal boundaries.

## Country outlines

`web/data/timezone-countries.json` contains 239 simplified country and territory geometries with names and ISO alpha-2 identifiers.

- Source: [datasets/geo-countries](https://github.com/datasets/geo-countries), pinned revision `ca3b7150caa5d3cc3fb108a87fc6e3d57bd97d99`, derived from Natural Earth 10m Admin-0 polygons. The dataset was simplified for the 900×520 overview map with Mapshaper `10% keep-shapes`.
- Size: 1,297,030 bytes. SHA-256: `fa790202c5a72dbad8f13f684af3ddca6f0874e0f006249358e7c70546387076`.
- Licensing: Natural Earth data is public domain; the distributing dataset is under ODbL/PDDL notices. See [Natural Earth terms](https://www.naturalearthdata.com/about/terms-of-use/) and the [pinned source](https://github.com/datasets/geo-countries/tree/ca3b7150caa5d3cc3fb108a87fc6e3d57bd97d99). Country names are joined with GeoNames country information; that source and its hash are recorded in the data file.

## Timezone regions

`web/data/timezone-zones.json` contains 419 IANA timezone polygons for world map clicks.

- Source: [Draw on a Map world timezone boundaries](https://drawonamap.com/datasets/world-time-zone-boundaries/), derived from [timezone-boundary-builder](https://github.com/evansiroky/timezone-boundary-builder) and OpenStreetMap contributors; source release 2026c, prepared 2026-09-15. The source is simplified at 0.02° tolerance and is explicitly approximate.
- Size: 1,913,514 bytes. SHA-256: `a9611bb721e4a0bf9607dbae10a5deb86e5f7b5f823bea314b13bd86b95af7f7`.
- License: ODbL 1.0, derived from OpenStreetMap contributors. The interface preserves manual IANA-zone entry/override for boundary edge cases. These polygons support location setup only; they do not establish legal boundaries.

## City index

`web/data/timezone-cities.json` bundles 34,188 GeoNames `cities15000` entries across 244 country codes. It retains city names, ASCII names, alternate names, administrative region, coordinates, population, feature code, and IANA zone. The country-name list contains 252 GeoNames country and territory names. City name and alias search resolves locally to coordinates and a named IANA zone; duplicate place names can be disambiguated with region-qualified suggestions.

- Source: [GeoNames export](https://download.geonames.org/export/dump/cities15000.zip), [administrative-region export](https://download.geonames.org/export/dump/admin1CodesASCII.txt), and [country information](https://download.geonames.org/export/dump/countryInfo.txt). Snapshot date: 2026-10-25.
- Size: 13,435,717 bytes. SHA-256: `2efc4794ff2a6c1ba6ed41352e9a3965a850218de92c186bd94c420dbd91d244`.
- Source archive SHA-256: `9c5e0684a0aeaa1b8a62ec08f2edcea94a425626806b8b249c1d9fc9b09168ec`; admin1 source SHA-256: `2e4867f5397acb056ea26377601b57af37cb4c81399770225018e9e439e91ad3`; country-info source SHA-256: `c55f638cfaafb4fe45a9aa15120a0f1f78e8a34a8e1847d7b9320cf740fbc9ab`.
- License: GeoNames data is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribution: GeoNames (https://www.geonames.org). Source hashes, snapshot details, and license are also stored in the JSON metadata.

The app uses the workstation's installed IANA timezone database for UTC-to-local and daylight-saving previews. Map and city data are bundled and available offline.
