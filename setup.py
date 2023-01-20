from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["pandas>=1", "numpy>=1"]

setup(
    name="snaplib",
    version="0.1.22",
    author="Artyom Kolas",
    author_email="artyom.kolas@gmail.com",
    description="Data preprocessing lib",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/kolasdevpy/snaplib",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)

