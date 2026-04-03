# Insolation


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15682056.svg)](https://doi.org/10.5281/zenodo.15682056)

This software consists of three Python libraries:

* astro.py: an interface (or computational module) providing various astronomical solutions (Berger 1978, Laskar 2004, ...) to deliver parameters needed for insolation: eccentricity, obliquity, and climatic precession. The reference solution is Laskar 2004.
* inso.py (depends on astro.py): computes different types of insolation—instantaneous, daily averaged, or other averaged insolation—as well as "caloric seasons", useful for paleoclimatic and long-term climate studies.
* minmax.py (depends on astro.py and inso.py): computes extremal values of daily insolation.

Some usage examples are also provided. In particular, `figures.py` generates several figures from the paper by Paillard (2026), *On the computation of several “insolation” quantities relevant to climatology or planetology*, published in *Climate of the Past*, 22, 647–673. https://doi.org/10.5194/cp-22-647-2026

```
import inso
inso.figures.display("1")
```
