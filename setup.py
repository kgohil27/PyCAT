"""
@author: Kanishk Gohil; kgohil@umd.edu
"""

from distutils.core import setup
setup(
  name = 'pyccntool',         # How you named your package folder (MyLib)
  packages = ['pyccntool'],   # Chose the same as "name"
  version = '1.0',      # Start with a small number and increase it with every change you make
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
        "re==2.2.1",
        "scikit-learn==0.21.3",
        "scipy==1.3.1",
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
