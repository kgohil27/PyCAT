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

MAJOR, MINOR = 1, 1
DEV = False
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

    fn = os.path.join(os.path.dirname(__file__), "pycat", "version.py")

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
  name = 'ccnalysis',        # How you named your package folder (MyLib)
  packages = ['pycat'],   # Chose the same as "name"
  version = VERSION,      # Start with a small number and increase it with every change you make
  license='New BSD (3-clause)',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'A package written in Python to perform CCN activity analysis.',   # Give a short description about your library
  author = 'Kanishk Gohil',                   # Type in your name
  author_email = 'kgohil@umd.edu',      # Type in your E-Mail
  url = 'https://github.com/kgohil27/PyCAT',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/kgohil27/PyCAT/archive/refs/tags/v1.0.tar.gz',    # I explain this later on
  install_requires=[            # I get to this in a second
        "more-itertools==7.2.0",
        "numba==0.45.1",
        "numpy==1.17.3",
        "pandas==0.25.2",
        "scikit-learn==0.21.3",
        "scipy==1.3.1",
        "seaborn==0.11.2",
        "xarray==0.12.3",
      ],
  classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
  ],
)
