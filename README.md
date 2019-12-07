![icon](docs/icon_20.png) 

# eddy-squeeze

Summarize and Visualize Information from FSL Eddy Outputs

```py
git clone https://github.com/pnlbwh/eddy-squeeze
```


## Contents

- TODO
- Dependencies
- How to use the script


## TODO

- move plot functions to the nifti-snapshot
- more information from eddy outputs


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



## Subject-wise summary of the eddy output

```shell

./eddy_squeeze --eddy_dir ../test/eddy_out -po

```


### Example outputs


|    |   restricted_absolute_motion |   restricted_relative_motion |   number_of_outlier_slices |   outlier_std_total |   outlier_std_mean |   outlier_std_std |
|----|------------------------------|------------------------------|----------------------------|---------------------|--------------------|-------------------|
|  0 |                     0.179872 |                     0.124634 |                         53 |             400.397 |            7.55465 |           2.61668 |



|   Outlier slices |   Volume |   Slice |      Stds |   Sqr_stds |   rank |
|------------------|----------|---------|-----------|------------|--------|
|                0 |       47 |      56 | -14.3535  |   38.5348  |      0 |
|                1 |       41 |      57 | -13.0163  |   34.9563  |      1 |
|                2 |       46 |      57 | -12.7129  |   33.6633  |      2 |
|                3 |       47 |      58 | -12.2244  |   39.2238  |      3 |
|                4 |       46 |      55 | -11.4638  |   29.2625  |      4 |
|                5 |       56 |      55 | -11.1451  |   23.2362  |      5 |
|                6 |       41 |      55 | -11.0186  |   21.3954  |      6 |
|                7 |       47 |      54 | -10.2093  |   21.6208  |      7 |
|                8 |       57 |      56 |  -9.91693 |   19.7956  |      8 |
|                9 |       51 |      56 |  -9.58872 |   19.8592  |      9 |
|               10 |       53 |      55 |  -9.58179 |   15.7306  |     10 |
|               11 |       56 |      57 |  -9.4811  |   15.7091  |     11 |
|               12 |       64 |      57 |  -8.95525 |   14.3134  |     12 |
|               13 |       64 |      55 |  -8.77195 |   13.5415  |     13 |
|               14 |       33 |      56 |  -8.6603  |   14.1161  |     14 |
|               15 |       53 |      53 |  -8.65345 |   13.3336  |     15 |
|               16 |       66 |      57 |  -8.47718 |   17.6086  |     16 |
|               17 |       65 |      52 |  -8.43979 |   21.6078  |     17 |


## Study-wise summary of the eddy outputs

```shell

./eddy_squeeze_study -ed /study/path/subject*/eddy

```

### Example output


|    | subject   |   number of volumes |   max b value |   min b value | unique b values                       |   number of b0s |   number of outlier slices |   Sum of standard deviations in outlier slices |   Mean of standard deviations in outlier slices |   Standard deviation of standard deviations in outlier slices |   absolute restricted movement |   relative restricted movement |   absolute movement |   relative movement |
|----|-----------|---------------------|---------------|---------------|---------------------------------------|-----------------|----------------------------|------------------------------------------------|-------------------------------------------------|---------------------------------------------------------------|--------------------------------|--------------------------------|---------------------|---------------------|
|  0 | subject_353c  |                  74 |          3000 |             0 | [   0.  200.  500. 1000. 2950. 3000.] |               5 |                         75 |                                        800.62  |                                        10.6749  |                                                       5.67277 |                       0.405855 |                       0.230262 |            0.653026 |            0.467002 |
|  1 | subject_309c  |                  74 |          3000 |             0 | [   0.  200.  500. 1000. 2950. 3000.] |               5 |                         62 |                                        585.877 |                                         9.44962 |                                                       5.26578 |                       0.728763 |                       0.112724 |            0.822696 |            0.412796 |
|  2 | subject_316c  |                  74 |          3000 |             0 | [   0.  200.  500. 1000. 2950. 3000.] |               5 |                         60 |                                        523.487 |                                         8.72478 |                                                       5.05003 |                       0.830369 |                       0.305694 |            0.952818 |            0.497189 |
|  3 | subject_369c  |                  74 |          3000 |             0 | [   0.  200.  500. 1000. 2950. 3000.] |               5 |                         58 |                                        580.126 |                                        10.0022  |                                                       5.76787 |                       0.345985 |                       0.159334 |            0.583275 |            0.739979 |
|  4 | subject_306c  |                  74 |          3000 |             0 | [   0.  200.  500. 1000. 2950. 3000.] |               5 |                         58 |                                        525.028 |                                         9.05221 |                                                       4.55145 |                       0.675001 |                       0.217329 |            0.801741 |            0.590544 |
|  5 | subject_275c  |                  74 |          3000 |             0 | [   0.  200.  500. 1000. 2950. 3000.] |               5 |                         54 |                                        508.574 |                                         9.41803 |                                                       4.52075 |                       0.59835  |                       0.271036 |            0.729778 |            0.509582 |
|  6 | subject_346c  |                  74 |          3000 |             0 | [   0.  200.  500. 1000. 2950. 3000.] |               5 |                         53 |                                        623.438 |                                        11.763   |                                                       5.57189 |                       0.444441 |                       0.50271  |            0.611553 |            0.647743 |
|  7 | subject_327c  |                  74 |          3000 |             0 | [   0.  200.  500. 1000. 2950. 3000.] |               5 |                         52 |                                        482.541 |                                         9.27963 |                                                       5.47736 |                       0.536185 |                       0.201369 |            0.652559 |            0.437693 |
|  8 | subject_330c  |                  74 |          3000 |             0 | [   0.  200.  500. 1000. 2950. 3000.] |               5 |                         51 |                                        427.729 |                                         8.38685 |                                                       4.72738 |                       0.838054 |                       0.711922 |            0.937807 |            0.827304 |
|  9 | subject_380c  |                  74 |          3000 |             0 | [   0.  200.  500. 1000. 2950. 3000.] |               5 |                         49 |                                        495.493 |                                        10.1121  |                                                       5.20017 |                       0.277204 |                       0.276305 |            0.490699 |            0.464468 |
| 10 | subject_356c  |                  74 |          3000 |             0 | [   0.  200.  500. 1000. 2950. 3000.] |               5 |                         49 |                                        386.715 |                                         7.89214 |                                                       3.56298 |                       0.313836 |                       0.247726 |            0.55314  |            0.448548 |


## Create figures to detect outliers

### example use in jupyter notebook

```py
from eddy_squeeze import kcho_eddy
eddy_pattern = '/path/to/study/subjects*/eddy'
eddyStudy = kcho_eddy.EddyStudy(eddy_pattern)
eddyStudy.clean_up_data_frame()

eddyStudy.df.sort_values(
    ['number of outlier slices', 'Sum of standard deviations in outlier slices', 
     'absolute restricted movement', 'relative restricted movement'],
    ascending=False
).drop(['ep', 'eddy_dir', 'unique b values', 'eddy_input'], axis=1).reset_index().drop('index', axis=1)


eddyStudy.df.groupby(['number of volumes', 'max b value', 'min b value', 'number of b0s']).count()['subject'].to_frame()
eddyStudy.get_unique_bvalues()
eddyStudy.figure_post_eddy_shell_PE()
eddyStudy.figure_post_eddy_shell()
eddyStudy.plot_subjects('absolute restricted movement')
eddyStudy.plot_subjects('relative restricted movement')
eddyStudy.plot_subjects('number of outlier slices')
eddyStudy.plot_subjects('Sum of standard deviations in outlier slices')
eddyStudy.plot_subjects('Mean of standard deviations in outlier slices')
eddyStudy.plot_subjects('Standard deviation of standard deviations in outlier slices')

```

## [Jupyter notebook example link](docs/eddy_summary_study_example.ipynb)



## Save outlier slices as figures

```shell
./eddy_squeeze --eddy_directory ../test/eddy_out
```

or

```shell
./eddy_squeeze --eddy_directory ../test/eddy_out --out_dir eddy_out_test
```


![output](docs/example_out.png)




