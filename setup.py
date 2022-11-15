from setuptools import setup, find_packages

install_requires = [
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "cmdstanpy",
    "holidays",
    "pytrends",
]

setup(
    name="bayesian_holidays",
    author="Daniel Marthaler",
    author_email="dan.marthaler@gmail.com",
    description="Bayesian Holiday Model for Time Series",
    url="https://github.com/mathDR/bayesian_holidays",
    packages=find_packages(exclude=["tests"]),
    test_suite="tests",
    install_requires=install_requires,
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
)
