# Create docs

The documentation site is built with [Jupyter Book](https://jupyterbook.org/).

Build it locally from the repository root:

```bash
make build-jupyterbook
```

Then open `jupyterbook/_build/html/index.html` to review it.

Deployment is automatic: on every merge to `main`, the `deploy` GitHub Actions
workflow builds the book and publishes it to GitHub Pages at
[spaceml-org.github.io/ml4floods](https://spaceml-org.github.io/ml4floods).

# Publish package to PyPI

First bump the version number in `ml4floods/__init__.py`, then from the
repository root:

```bash
make build      # build sdist + wheel (uv build)
make publish    # upload to PyPI (uv publish)
```

The package is listed at [pypi.org/project/ml4floods](https://pypi.org/project/ml4floods/).
