![icon](docs/icon_20.png) 

# eddy-squeeze

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3600531.svg)](https://doi.org/10.5281/zenodo.3600531)



Summarize and Visualize Information from FSL Eddy Outputs

```sh
eddy-squeeze/bin/eddy_squeeze \
    --eddy_directories \
        /eddy/study/dir/subject01 \
        /eddy/study/dir/subject02 \
        /eddy/study/dir/subject03 \
    --out_dir prac_eddy_summary \
    --print_table \
    --figure \
    --save_html
```



## Contents

- Installation
- Dependencies
- How to use the script


## Installation

```sh
git clone https://github.com/pnlbwh/eddy-squeeze
```

```sh
cd eddy-squeeze/bin
./eddy_squeeze -h
```


## Dependencies

```py
FSL 6.0.1 EDDY outputs
FSLDIR in PATH
nifti-snapshot (https://github.com/pnlbwh/nifti-snapshot)
python 3.7
scipy==1.3.3
nibabel==2.4.0
numpy==1.16.2
pathlib2==2.3.3
matplotlib==3.0.3
tabulate==0.8.5
```



## Summary of the eddy outputs


### Print only
```sh
# one eddy output
eddy_squeeze --eddy_directories /test/eddy_out --print_table

# two eddy outputs
eddy_squeeze --eddy_directories /test/eddy_out1 /test/eddy_out2 --print_table
```


### Save html summary

```sh
eddy_squeeze --eddy_directories /test/eddy_out1 /test/eddy_out2 \
    --print_table \
    --save_html \
    --out_dir prac_eddy_summary
```


### Save html summary with figures

```sh
eddy_squeeze --eddy_directories /test/eddy_out1 /test/eddy_out2 \
    --print_table \
    --save_html \
    --figure \
    --out_dir prac_eddy_summary
```



### Example outputs

```sh
eddy_squeeze --eddy_directories ../test/eddy_out --print_table
```

```sh

Output directory : /Users/kevin/eddy-squeeze/tests/bin/prac_eddy_summary
--------------------------------------------------

Setting up eddy directories
--------------------------------------------------

Extracting information from all eddy outputs
--------------------------------------------------
Summarizing 3 subjects
There is no eddy related files in ../prac_study_dir/subject03

n=2 eddy outputs detected
--------------------------------------------------

Basic information
--------------------------------------------------
+----+-----------+-----------------------------------------------------------------+---------------------+---------------+---------------+---------------------------------------+-----------------+
|    | subject   | eddy_dir                                                        |   number of volumes |   max b value |   min b value | unique b values                       |   number of b0s |
|----+-----------+-----------------------------------------------------------------+---------------------+---------------+---------------+---------------------------------------+-----------------|
|  0 | subject01 | /Users/kevin/eddy-squeeze/tests/bin/../prac_study_dir/subject01 |                  74 |          3000 |             0 | [   0.  200.  500. 1000. 2950. 3000.] |               5 |
|  1 | subject02 | /Users/kevin/eddy-squeeze/tests/bin/../prac_study_dir/subject02 |                  74 |          3000 |             0 | [   0.  200.  500. 1000. 2950. 3000.] |               5 |
+----+-----------+-----------------------------------------------------------------+---------------------+---------------+---------------+---------------------------------------+-----------------+

Outlier information
--------------------------------------------------
+----+-----------+----------------------------+------------------------------------------------+-------------------------------------------------+---------------------------------------------------------------+
|    | subject   |   number of outlier slices |   Sum of standard deviations in outlier slices |   Mean of standard deviations in outlier slices |   Standard deviation of standard deviations in outlier slices |
|----+-----------+----------------------------+------------------------------------------------+-------------------------------------------------+---------------------------------------------------------------|
|  0 | subject01 |                         80 |                                        894.399 |                                           11.18 |                                                       6.30107 |
|  1 | subject02 |                         80 |                                        894.399 |                                           11.18 |                                                       6.30107 |
+----+-----------+----------------------------+------------------------------------------------+-------------------------------------------------+---------------------------------------------------------------+

Motion information
--------------------------------------------------
+----+-----------+--------------------------------+--------------------------------+
|    | subject   |   absolute restricted movement |   relative restricted movement |
|----+-----------+--------------------------------+--------------------------------|
|  0 | subject01 |                       0.190404 |                       0.112074 |
|  1 | subject02 |                       0.190404 |                       0.112074 |
+----+-----------+--------------------------------+--------------------------------+

```



![mainEddy](docs/eddy_summary_main.png)

![subjectEddy](docs/eddy_summary_subject.png)

![output](docs/example_out.png)
