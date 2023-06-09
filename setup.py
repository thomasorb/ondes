from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages
import io
import codecs
import os
import sys
import numpy

packages = find_packages(where=".")

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    
extensions = [
    Extension(
        "ondes.ccore",
        [
            "ondes/ccore.pyx"
        ],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "ondes.cserver",
        [
            "ondes/cserver.pyx"
        ],
        include_dirs=[numpy.get_include()]
    ),
]

setup(
    name='ondes',
    ext_modules=cythonize(extensions, annotate=True),
    version='1.0',
    url='https://myurl.com',
    license='GPLv3+',
    author='Thomas Martin',
    author_email='thomas.martin.1@ulaval.ca',
    maintainer='Thomas Martin',
    maintainer_email='thomas.martin.1@ulaval.ca',
    setup_requires=['cython', 'numpy'],
    description='Cython example',
    long_description=long_description,
    packages=packages,
    package_dir={"": "."},
    include_package_data=True,
    package_data={
        '':['LICENSE.txt', '*.rst', '*.txt', 'docs/*', '*.pyx'],
        'ondes':['data/*', '*.pyx']},
    exclude_package_data={
        '': ['*~', '*.so', '*.pyc', '*.c'],
        'ondes':['*~', '*.so', '*.pyc', '*.c']},
    platforms='any',
    scripts=[],
    classifiers = [
        'Programming Language :: Python',
        'Programming Language :: Cython',
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent' ],
)
