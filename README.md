![icon](docs/icon_20.png) 

# eddy-squeeze

Visualize extra information from FSL eddy outputs

```py
git clone https://github.com/pnlbwh/eddy-squeeze
```


## Contents

- TODO
- Dependencies
- How to use the script


## TODO

- move plot functions to the nifti-snapshot
- clean up the script
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
```

## How to use the script

### Save outlier slices as figures
```shell
./eddy_squeeze --eddy_directory ../test/eddy_out
```

or

```shell
./eddy_squeeze --eddy_directory ../test/eddy_out --out_dir eddy_out_test
```


![output](docs/example_out.png)



### Print motion (restricted) along with other eddy information

```shell
./eddy_squeeze --eddy_dir ../test/eddy_out -po
```

```
restricted_absolute_motion  restricted_relative_motion  number_of_outlier_slices  outlier_std_total  outlier_std_mean  outlier_std_std
0                    0.179872                    0.124634                        53          400.39664          7.554654         2.616683
                Volume  Slice      Stds  Sqr_stds  rank

Outlier slices
0                   47     56 -14.35350  38.53480     0
1                   41     57 -13.01630  34.95630     1
2                   46     57 -12.71290  33.66330     2
3                   47     58 -12.22440  39.22380     3
4                   46     55 -11.46380  29.26250     4
5                   56     55 -11.14510  23.23620     5
6                   41     55 -11.01860  21.39540     6
7                   47     54 -10.20930  21.62080     7
8                   57     56  -9.91693  19.79560     8
9                   51     56  -9.58872  19.85920     9
10                  53     55  -9.58179  15.73060    10
11                  56     57  -9.48110  15.70910    11
12                  64     57  -8.95525  14.31340    12
13                  64     55  -8.77195  13.54150    13
14                  33     56  -8.66030  14.11610    14
15                  53     53  -8.65345  13.33360    15
16                  66     57  -8.47718  17.60860    16
17                  65     52  -8.43979  21.60780    17
18                  47     52  -8.43364  20.26750    18
19                  25     56  -8.31707  16.72230    19
20                  51     58  -8.26085  16.33100    20
21                  56     53  -8.24303  18.00460    21
22                  48     57  -8.14592  16.64700    22
23                  47     50  -8.09716  18.92510    23
24                  65     50  -7.91556  18.84600    24
25                  61     56  -7.79518  14.29030    25
26                  53     57  -7.74794  10.99030    26
27                  36     55  -7.67407  12.38540    27
28                  36     57  -7.51060  11.60880    28
29                  65     54  -7.36123  12.12250    29
30                  53     51  -7.25939  14.14500    30
31                  62     44  -6.93178  16.83770    31
32                  57     54  -6.77710  10.85570    32
33                  25     52  -6.73062  12.48860    33
34                  66     55  -6.57132   9.55288    34
35                  39     45  -6.55768   8.46479    35
36                  33     58  -6.49373  11.84110    36
37                  57     58  -5.31691   7.15539    37
38                  61     58  -4.88429   6.36768    38
39                  68     57  -4.80080   4.67743    39
40                  43     57  -4.63532   3.86947    40
41                  65     48  -4.55496   4.42666    41
42                  43     55  -4.53226   4.62069    42
43                  44     56  -4.46189   3.35124    43
44                  64     53  -4.46001   3.86975    44
45                  46     53  -4.43432   6.07274    45
46                  48     22  -4.39074   5.49077    46
47                  44     58  -4.18749   4.05597    47
48                  65     46  -4.06953   4.31454    48
49                  48     55  -4.05850   3.87897    49
50                  25     54  -4.05688   5.12469    50
51                  56     51  -4.04159   5.84227    51
52                  46     51  -4.01692   5.64939    52
```

