from typing import List

from setuptools import find_namespace_packages, setup


with open("README.md", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup_requirements: List[str] = []

requirements = [
    "f90nml>=1.1.0",
    "numpy==1.26.4",
    "xarray",
]

test_requirements = ["pytest", "pytest-subtests", "serialbox", "coverage"]
ndsl_requirements = ["ndsl @ git+https://github.com/NOAA-GFDL/NDSL.git@develop"]
pyFV3_requirements = ["pyFV3 @ git+https://github.com/NOAA-GFDL/PyFV3.git@develop"]
develop_requirements = (
    test_requirements + ndsl_requirements + pyFV3_requirements + ["pre-commit"]
)

extras_requires = {
    "test": test_requirements,
    "ndsl": ndsl_requirements,
    "pyFV3": pyFV3_requirements,
    "develop": develop_requirements,
}


setup(
    author="NOAA - Geophysical Fluid Dynamics Laboratory",
    author_email="oliver.elbert@noaa.gov",
    python_requires=">=3.11.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description="pySHiELD is a collection of NDSL-based physical parameterizations"
    "for atmospheric models",
    install_requires=requirements,
    extras_require=extras_requires,
    license="Apache 2.0 license",
    long_description=readme,
    include_package_data=True,
    name="pySHiELD",
    packages=find_namespace_packages(include=["pySHiELD", "pySHiELD.*"]),
    setup_requires=[],
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/NOAA-GFDL/PySHiELD",
    version="0.2.0",
    zip_safe=False,
)
