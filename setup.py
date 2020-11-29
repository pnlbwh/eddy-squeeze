import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="eddy-squeeze-kcho", # Replace with your own username
    version="2.0.7",
    author="Kevin Cho",
    author_email="kevincho@bwh.harvard.edu",
    description="Visualize extra information from FSL 6.0.1 eddy outputs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pnlbwh/eddy-squeeze",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.6',
    install_requires=[
        'tabulate>=0.8',
        'pathlib',
        'argparse',
        'pandas',
        'numpy',
        'seaborn',
        'matplotlib',
        'nibabel',
        'jinja2'
        ],
    scripts=['bin/eddy_squeeze']
)
