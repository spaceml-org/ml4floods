.PHONY: help install install-dev sync lock format lint type test check clean build publish build-jupyterbook clean-jupyterbook publish-docs
.DEFAULT_GOAL = help

PKGROOT = ml4floods
PYTHON_VERSION = 3.11

help:	## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Environment

install:  ## Install package into a uv-managed venv
	uv sync

install-dev:  ## Install package with dev/tests/docs extras
	uv sync --extra dev --extra tests --extra docs

sync:  ## Sync env to lockfile
	uv sync --extra dev --extra tests --extra docs

lock:  ## Refresh uv.lock
	uv lock

##@ Formatting & Linting

format:  ## Format code with ruff
	uv run ruff format $(PKGROOT) tests

lint:  ## Lint code with ruff (auto-fix safe issues)
	uv run ruff check --fix $(PKGROOT) tests

##@ Type Checking

type:  ## Type check with mypy
	uv run mypy $(PKGROOT)

##@ Testing

test:  ## Run pytest
	uv run pytest -v tests

check: lint type test  ## Lint + type-check + test

##@ Docs

build-jupyterbook:  ## Build jupyter book
	uv run jupyter-book build jupyterbook --all

clean-jupyterbook:  ## Clean jupyter book html
	uv run jupyter-book clean jupyterbook

publish-docs:  ## Publish docs to gh-pages
	uv run ghp-import -n -p -f jupyterbook/_build/html

##@ Build & Publish

clean:  ## Remove build artifacts
	rm -rf build dist *.egg-info

build:  ## Build sdist + wheel
	uv build

publish:  ## Publish to PyPI
	uv publish
