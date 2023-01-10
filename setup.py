

from setuptools import setup, find_packages
from os.path import join, dirname
import snaplib.snaplib as snaplib

setup(
    entry_points = {'console_scripts': ['trash = snaplib.snaplib:main']},
    name ='snaplib',
    author = 'Artyom Kolas', 
    author_email = 'artyom.kolas@gmail.com',
    license = 'MIT',
    version = '0.0.1',
    packages=find_packages(),
    long_description=open(join(dirname(__file__), 'README.md')).read(),
)