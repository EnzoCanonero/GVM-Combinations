from setuptools import setup, find_packages

setup(
    name="gvm",
    version="0.1.0",
    description="Gamma Variance Model for combining correlated measurements",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/EnzoCanonero/errors-on-errors",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "PyYAML",
        "iminuit",
        "scipy",
        "matplotlib",
    ],
)
