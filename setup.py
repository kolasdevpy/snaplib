from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = [
                "pandas==1.5.3", 
                "numpy==1.24.0", 
                "matplotlib==3.7.0", 
                "lightgbm==3.3.5", 
                "sklearn==1.2.1", 
                "seaborn==0.12.2", 
                ]

setup(
    name="snaplib",
    version="0.4.6",
    author="Artsiom Kolas",
    author_email="artyom.kolas@gmail.com",
    description="Data preprocessing lib",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/kolasdevpy/snaplib",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)

