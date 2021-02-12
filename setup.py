from setuptools import setup, find_packages


def parse_requirements_file(filename):
    with open(filename, encoding="utf-8") as fid:
        requires = [l.strip() for l in fid.readlines() if l]
    return requires


# Optional Packages
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

setup(
    name="ml4floods",
    version="0.0.1",
    author="SpaceML-org",
    # author_email="t.pinder2@lancaster.ac.uk",
    packages=find_packages(".", exclude=["tests"]),
    # license="LICENSE",
    description="Machine learning methods for floods.",
    # long_description="",
    install_requires=parse_requirements_file("requirements.txt"),
    extras_require=EXTRAS,
    keywords=["floods pytorch machine-learning earth"],
)