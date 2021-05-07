from setuptools import setup, find_packages


def parse_requirements_file(filename):
    with open(filename, encoding="utf-8") as fid:
        requires = [l.strip() for l in fid.readlines() if l]
    return requires


# Optional Packages
# See https://godatadriven.com/blog/a-practical-guide-to-using-setup-py/
EXTRAS = {
    "dev": ["black", "isort", "pylint", "flake8", "pyprojroot"],
    "tests": [
        "pytest",
    ],
    "docs": [
        "furo==2020.12.30b24",
        "nbsphinx==0.8.1",
        "nb-black==1.0.7",
        "matplotlib==3.3.3",
        "sphinx-copybutton==0.3.1",
    ],
}

setup(name="ml4floods",
      version="0.0.1",
      author="SpaceML-org",
      packages=find_packages(".", exclude=["tests"]),
      package_data={
        "ml4floods" : ["models/configurations/*.json"]  # Add json files from configuration dir.
      },
      description="Machine learning methods for flood extent segmentation.",
      install_requires=parse_requirements_file("requirements.txt"),
      extras_require=EXTRAS,
      keywords=["floods pytorch machine-learning earth"],
)