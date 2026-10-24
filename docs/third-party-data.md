# Bundled geographic data

`web/data/timezone-countries.json` contains the country boundaries used by the offline mission-location map.

- Source: [datasets/geo-countries](https://github.com/datasets/geo-countries), pinned revision `ca3b7150caa5d3cc3fb108a87fc6e3d57bd97d99`, derived from Natural Earth 10m Admin-0 country polygons.
- Scope: Iran, Lebanon, Israel, Latvia, and Finland only. The file retains each selected geometry, country name, ISO alpha-2 code, and the associated IANA zone.
- Size: 159,435 bytes. SHA-256: `6387aec692371472bdeee100194607ccaac4543b6efd446e090bea15b2200758`.
- Licensing: Natural Earth places its country data in the public domain. The distributing `datasets/geo-countries` project uses the Open Data Commons Public Domain Dedication and License (PDDL) 1.0. The bundled file is a filtered derivative of those polygons. See [Natural Earth's terms](https://www.naturalearthdata.com/about/terms-of-use/) and the [source repository](https://github.com/datasets/geo-countries/tree/ca3b7150caa5d3cc3fb108a87fc6e3d57bd97d99).
- Boundary note: Natural Earth depicts de facto country boundaries. They are illustrative map geometry; operators can edit the country and IANA zone fields after selecting a point.
- Zone mapping: The five country-to-IANA-zone identifiers were checked against the staged IANA `zone.tab` in `tzdata` 2026.5. The app previews a UTC instant using the selected IANA zone and the runtime timezone database. The map is scoped to these supported countries; it is not a worldwide timezone-boundary dataset.

The map is served as a local application asset and makes no tile, geocoding, or timezone API requests.
