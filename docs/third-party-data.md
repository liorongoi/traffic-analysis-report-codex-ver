# Bundled geographic data

`web/data/timezone-countries.json` contains the country boundaries used by the offline mission-location map.

- Source: [datasets/geo-countries](https://github.com/datasets/geo-countries), pinned revision `ca3b7150caa5d3cc3fb108a87fc6e3d57bd97d99`, derived from Natural Earth 10m Admin-0 country polygons.
- Scope: Iran, Lebanon, Israel, Latvia, and Finland only. The file retains each selected geometry, country name, ISO alpha-2 code, and the associated IANA zone.
- Size: 159,435 bytes. SHA-256: `6387aec692371472bdeee100194607ccaac4543b6efd446e090bea15b2200758`.
- Licensing: Natural Earth places its country data in the public domain. The distributing `datasets/geo-countries` project uses the Open Data Commons Public Domain Dedication and License (PDDL) 1.0. The bundled file is a filtered derivative of those polygons. See [Natural Earth's terms](https://www.naturalearthdata.com/about/terms-of-use/) and the [source repository](https://github.com/datasets/geo-countries/tree/ca3b7150caa5d3cc3fb108a87fc6e3d57bd97d99).
- Boundary note: Natural Earth depicts de facto country boundaries. They are illustrative map geometry; operators can edit the country and IANA zone fields after selecting a point.
- Zone mapping: The five country-to-IANA-zone identifiers were checked against the staged IANA `zone.tab` in `tzdata` 2026.5. The app previews a UTC instant using the selected IANA zone and the runtime timezone database. The map polygons are scoped to these five supported countries; this is not a worldwide boundary map. The UI can search IANA zone names supplied by the workstation’s installed Python timezone database through the local `/api/timezones` endpoint.

## City index

`web/data/timezone-cities.json` bundles the 670 GeoNames `cities15000` records with an IANA zone for Israel (106), Iran (429), Lebanon (18), Latvia (15), and Finland (102). It retains the city name, ASCII name, alternate names, administrative-region name, coordinates, population, and feature code. City names and aliases resolve locally to coordinates and a named IANA zone; duplicate names are disambiguated by region. The index does not add map polygons outside the five supported countries.

- Source: [GeoNames export](https://download.geonames.org/export/dump/cities15000.zip) and [GeoNames administrative-region export](https://download.geonames.org/export/dump/admin1CodesASCII.txt). Snapshot date: 2026-10-25.
- Source archive SHA-256: `9c5e0684a0aeaa1b8a62ec08f2edcea94a425626806b8b249c1d9fc9b09168ec`; administrative-region file SHA-256: `2e4867f5397acb056ea26377601b57af37cb4c81399770225018e9e439e91ad3`.
- License: GeoNames data is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribution: GeoNames (https://www.geonames.org). The source archive and index record are provided as-is; the index is filtered to the five project-supported countries.

The map and city index are served as local application assets and make no tile, geocoding, or timezone API requests.
