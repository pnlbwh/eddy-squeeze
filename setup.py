import setuptools
from os.path import join

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="eddy-squeeze", # Replace with your own username
    version="1.0.7",
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
    install_requires=['tabulate>=0.8.7',
                      'matplotlib>=3.0.3',
                      'pandas>=0.24',
                      'numpy>=1.1',
                      'nibabel>=3.1',
                      'Jinja2>=2.0',
                      'pathlib2>=2.3',
                      'seaborn>=0.9'],
    scripts=['scripts/eddy_squeeze']
)

