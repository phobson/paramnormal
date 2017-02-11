# Setup script for the paramnormal package
#
# Usage: python setup.py install

import os
from setuptools import setup, find_packages


DESCRIPTION = "paramnormal: Conventionally parameterized probability distributions"
LONG_DESCRIPTION = DESCRIPTION
NAME = "paramnormal"
VERSION = "v0.3.0"
AUTHOR = "Paul Hobson"
AUTHOR_EMAIL = "pmhobson@gmail.com"
URL = 'http://phobson.github.io/paramnormal/'
DOWNLOAD_URL = "https://github.com/phobson/paramnormal/archive/master.zip"
LICENSE = "BSD 3-clause"
PACKAGES = find_packages()
PLATFORMS = "Python 3.4 and later."
CLASSIFIERS = [
    'Programming Language :: Python',
    'Operating System :: OS Independent',
    'License :: OSI Approved :: BSD License',
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
]
INSTALL_REQUIRES = ['numpy', 'scipy', 'matplotlib']
PACKAGE_DATA = {
    'paramnormal.tests.baseline_images.test_activity': ['*png'],
}
DATA_FILES = []


if __name__ == "__main__":
    setup(
        name=NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        download_url=DOWNLOAD_URL,
        license=LICENSE,
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        data_files=DATA_FILES,
        platforms=PLATFORMS,
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
        zip_safe=False
    )
