from pathlib import Path

from setuptools import find_packages, setup

_PLAYGROUND = Path(__file__).resolve().parent.parent

setup(
    name="rm75-control",
    version="0.1.0",
    description="RealMan RM75 integrated controller wrapper",
    packages=find_packages(include=["rm75_control*"])
    + find_packages(where=str(_PLAYGROUND), include=["peirastic*"]),
    package_dir={"peirastic": str(_PLAYGROUND / "peirastic")},
    package_data={"peirastic": ["configs/*.yaml"]},
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=["numpy", "pyyaml", "ruckig==0.17.3"],
)
