.PHONY: conda conda_dev envupdate envupdatedev format style types black sort test link check
.DEFAULT_GOAL = help

PYTHON = python
VERSION = 3.8
NAME = py_name
ROOT = ./
PIP = pip
CONDA = conda
SHELL = bash
PKGROOT = ml4floods

help:	## Display this help
		@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Install Environments

conda:  ## setup a conda environment
		$(info Installing the environment)
		@printf "Creating conda environment...\n"
		${CONDA} create env create -f environment.yml
		@printf "\n\nConda environment created! \033[1;34mRun \`conda activate ${NAME}\` to activate it.\033[0m\n\n\n"

conda_dev:  ## setup a conda environment for development
		@printf "Creating conda dev environment...\n"
		${CONDA} create env create -f environment_dev.yml
		@printf "\n\nConda dev environment created! \033[1;34mRun \`conda activate ${NAME}\` to activate it.\033[0m\n\n\n"

##@ Update Environments

envupdate: ## update conda environment
		@printf "Updating conda environment...\n"
		${CONDA} env update -f environment.yml
		@printf "Conda environment updated!"
	
envupdatedev: ## update conda environment
		@printf "Updating conda dev environment...\n"
		${CONDA} env update -f environment_dev.yml
		@printf "Conda dev environment updated!"

##@ Formatting

black:  ## Format code in-place using black.
		black ${PKGROOT}/ tests/ -l 79 .

sort:   ## Format code in-place using black.
		isort ${PKGROOT}/ tests/ -l 79 .

format: ## Code styling - black, isort
		black --check --diff ${PKGROOT} tests
		@printf "\033[1;34mBlack passes!\033[0m\n\n"
		isort ${PKGROOT}/ tests/
		@printf "\033[1;34misort passes!\033[0m\n\n"

style:  ## Code lying - pylint
		@printf "Checking code style with flake8...\n"
		flake8 ${PKGROOT}/
		@printf "\033[1;34mPylint passes!\033[0m\n\n"
		@printf "Checking code style with pydocstyle...\n"
		pydocstyle ${PKGROOT}/
		@printf "\033[1;34mpydocstyle passes!\033[0m\n\n"

lint: format style types  ## Lint code using pydocstyle, black, pylint and mypy.
check: lint test  # Both lint and test code. Runs `make lint` followed by `make test`.

##@ Type Checking

types:	## Type checking with mypy
		@printf "Checking code type signatures with mypy...\n"
		python -m mypy ${PKGROOT}/
		@printf "\033[1;34mMypy passes!\033[0m\n\n"

##@ Testing

test:  ## Test code using pytest.
		@printf "\033[1;34mRunning tests with pytest...\033[0m\n\n"
		pytest -v tests
		@printf "\033[1;34mPyTest passes!\033[0m\n\n"

##@ DOCS

build-jupyterbook:  ## build jupyter book documentation
	jupyter-book build jupyterbook --all

clean-jupyterbook:  ## clean the jupyter book html files
	jupyter-book clean jupyterbook

build-package:
	rm -rf build/
	rm -rf dist/
	python setup.py sdist bdist_wheel

publish-package:
	python -m twine upload dist/*

publish-docs:
	ghp-import -n -p -f jupyterbook/_build/html
