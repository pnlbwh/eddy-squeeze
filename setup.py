import setuptools
from os.path import join

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements= open('requirements.txt').read().split()

setuptools.setup(
    name="eddy-squeeze", # Replace with your own username
    version="1.0.4",
    author="Kevin Cho",
    author_email="kevincho@bwh.harvard.edu",
    description="Visualize extra information from FSL 6.0.1 eddy outputs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pnlbwh/eddy-squeeze",
    packages=setuptools.find_packages(),
    package_data={'eddy_squeeze':[
        join('html_templates','*'),
        ]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.7',
    install_requires=requirements,
    scripts=['bin/eddy_squeeze']
)
