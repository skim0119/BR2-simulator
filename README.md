<div align="center">
<h1> BR2 Simulator </h1>

<img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white"/>
<a href='https://br2-simulator.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/br2-simulator/badge/?version=latest' alt='Documentation Status'/>
</a>

</div>

----

> Under modification to accomodate new updates in `PyElastica`, and adding verification results for AIYS2026

Simulator development for BR2 Softrobot using [`PyElastica`](https://github.com/GazzolaLab/PyElastica).

The [documentation](https://br2-simulator.readthedocs.io/en/latest/?badge=latest) includes the detail guide of how to use this package.

## Development

Python 3.11 and [uv](https://docs.astral.sh/uv/) are required. Pillow 13's
security fixes require Python 3.11; Python 3.8–3.10 are no longer supported.

```sh
uv sync --locked
```

Install documentation dependencies and build locally with:

```sh
uv sync --locked --group docs
uv run sphinx-build -b html docs docs/_build/html
```

## Configuration

The Sample rod-database is included in [`sample_database`](https://github.com/skim0119/BR2-simulator/tree/main/sample_database) and the sample assembly is included in [`sample_assembly`](https://github.com/skim0119/BR2-simulator/tree/main/sample_assembly).

## Interactive BR2

_work in progress_

- run_br2_interactive.py

