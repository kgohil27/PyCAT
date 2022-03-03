"""
@author: Kanishk Gohil; kgohil@umd.edu
"""

try:
    from setuptools import setup

    HAS_SETUPTOOLS = True
except ImportError:
    from distutils.core import setup

import os
import warnings
from textwrap import dedent

MAJOR, MINOR = 1, 0
DEV = True
VERSION = "{}.{}".format(MAJOR, MINOR)

# Correct versioning with git info if DEV
if DEV:
    import subprocess

    pipe = subprocess.Popen(
        ["git", "describe", "--always", "--match", "'v[0-9]*'"],
        stdout=subprocess.PIPE,
    )
    so, err = pipe.communicate()

    if pipe.returncode != 0:
        # no git or something wrong with git (not in dir?)
        warnings.warn(
            "WARNING: Couldn't identify git revision, using generic version string"
        )
        VERSION += ".dev"
    else:
        git_rev = so.strip()
        git_rev = git_rev.decode("ascii")  # necessary for Python >= 3

        VERSION += ".dev-" + format(git_rev)

extensions = [
    # Numba AOT extension module
    # auxcc.distutils_extension(),
]


def _write_version_file():

    fn = os.path.join(os.path.dirname(__file__), "PyCAT", "version.py")

    version_str = dedent(
        """
        __version__ = '{}'
        """
    )

    # Write version file
    with open(fn, "w") as version_file:
        version_file.write(version_str.format(VERSION))


# Write version and install
_write_version_file()

setup(
    name="PyCAT",
    author="Kanishk Gohil",
    author_email="kgohil@umd.edu",
    maintainer="Kanishk Gohil",
    maintainer_email="kgohil@umd.edu",
    description="A package written in Python to perform CCN activity analysis.",
    long_description="""
        PyCAT is a python package developed for CCN activity analysis of aerosols. PyCAT
        combines the time series size-resolved measurements of total aerosol counts and
        droplet counts obtained from typical CCN experimental setups. These measurements
        can then be used for either calibrating the CCNC supersaturation, or for generating
        the analysis data for any given aerosol. In the current form, PyCAT is capable of
        generating analysis data using the Traditional Kohler theory. In the subsequent
        builds, additional models will be added to the code.
    """,
    license="New BSD (3-clause)",
    url="https://github.com/kgohil27/PyCAT",
    version=VERSION,
    download_url="https://github.com/kgohil27/PyCAT",
    # TODO: Update install requirements and corresponding documentation
    install_requires=[
        "more-itertools==7.2.0",
        "numba==0.45.1",
        "numpy==1.17.3",
        "pandas==0.25.2",
        "re==2.2.1",
        "scikit-learn==0.21.3",
        "scipy==1.3.1",
        "xarray==0.12.3",
    ],
    packages=["PyCAT"],
    ext_modules=extensions,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD 3-Clause License",
        "Natural Language :: English",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
)
