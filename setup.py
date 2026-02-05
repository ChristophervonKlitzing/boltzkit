from codecs import open
from os import path

from setuptools import find_packages, setup

__version__ = "0.1"

here = path.abspath(path.dirname(__file__))

# Get the dependencies and installs:
with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    all_reqs = f.read().split("\n")
install_requires = [x.strip() for x in all_reqs]

setup(
    name="boltzkit",
    version=__version__,
    description="Boltzmann Kit",
    url="https://github.com/ChristophervonKlitzing/boltzkit",
    keywords="",
    packages=find_packages(),
    include_package_data=True,
    author=["Denis Blessing, Henrik Schopmans, Christopher von Klitzing"],
    install_requires=install_requires,
)
