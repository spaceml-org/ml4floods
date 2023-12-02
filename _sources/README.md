# Create docs

Install `jupyter-book` and `ghp-import`. From the main directory run

```bask
make build-jupyterbook
# Check doc in local looks good

# Upload the doc to GitHub
ghp-import -n -p -f jupyterbook/_build/html
```
The commit should appear in branch `gh-pages` and the page will be live at [spaceml-org.github.io/ml4floods](https://spaceml-org.github.io/ml4floods)

# Publish package in pip

Install `twine`.

First update the version number in `ml4floods/__init__.py`

```
rm -rf build/
rm -rf dist/
python setup.py sdist bdist_wheel

# upload to testpypi
python -m twine upload --repository testpypi dist/*

# Upload to real pypi
python -m twine upload dist/*
```
The package in PyPi will be shown at [pypi.org/project/ml4floods/](https://pypi.org/project/ml4floods/). The test version can be checked at [test.pypi.org/project/ml4floods/](https://test.pypi.org/project/ml4floods/) 


Follow [this tutorial](https://towardsdatascience.com/how-to-publish-a-python-package-to-pypi-7be9dd5d6dcd) to understand how pip works
