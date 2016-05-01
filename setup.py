#!/usr/bin/env python

from setuptools import setup

from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

setup(name='ConvNetPy',
      version='1.0',
      description='Python port of ConvNetJS',
      long_description=long_description,
      license='MIT',
      url='https://github.com/Aaronduino/ConvNetPy',
      packages=['convnetpy', 'convnetpy.layers'],
     )
