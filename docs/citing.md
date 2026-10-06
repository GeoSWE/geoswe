# Citing GeoSWE

If you use GeoSWE in your research, please cite it. Citation metadata is in
[`CITATION.cff`](https://github.com/GeoSWE/geoswe/blob/main/CITATION.cff) at the
repository root (GitHub renders a "Cite this repository" button from it).

```bibtex
@software{geoswe,
  title   = {{GeoSWE}: Geophysical Shallow-Water Engine},
  author  = {Chen, Peng},
  year    = {2026},  version = {1.1.1},
  doi     = {10.5281/zenodo.23198244},
  license = {BSD-3-Clause},
  url     = {https://github.com/GeoSWE/geoswe}
}
```

The DOI above is the *concept* DOI: it always resolves to the newest version, which is
what you want when citing the software in general. Every release also has its own DOI,
and that is the one to cite for work whose results depend on it; version 1.1.1 is
[10.5281/zenodo.23198245](https://doi.org/10.5281/zenodo.23198245), and the Zenodo page
lists the rest.

Cite the version you ran, not "the latest". Releases change what the solver computes:
1.1.0 fixed two defects that produced plausible but wrong results in 1.0.0, so "GeoSWE"
without a version does not identify what produced a number. Pin it in the environment
that produced the results (`pip install geoswe==1.1.1`), put that version in the
`version` field above, and link the matching documentation, which Read the Docs keeps
per version at `https://geoswe.readthedocs.io/en/v1.1.1/`.

A paper describing the method, the compressed active-cell mesh, the cross-code benchmark against TRITON, SynxFlow, and SERGHEI, and the county-to-continental applications is in preparation; this
page will be updated with the article citation when it is available.

## License

GeoSWE is distributed under the **BSD 3-Clause** license; see
[`LICENSE`](https://github.com/GeoSWE/geoswe/blob/main/LICENSE).
