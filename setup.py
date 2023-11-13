# pip install -e . (from ste directory)

# pip3 cache purge
# pip3 install -r requirements.txt --no-cache-dir

# Actually: TMPDIR=../../tmp/ pip3 install -e . --no-cache-dir

from setuptools import setup, find_packages

setup(
    name="ste_NLDL",
    version="0.0.1",
    author="alouettevanhove",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "black",
        "pycodestyle",
        "pytest",
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        "pyyaml"
    ],
)
