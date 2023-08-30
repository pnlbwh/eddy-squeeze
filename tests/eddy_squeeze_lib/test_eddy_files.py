from pathlib import Path
import os
import pandas as pd
import numpy as np
import sys
import pytest

script_loc = Path(os.path.realpath(__file__))
script_dir = script_loc.parent
test_root_dir = script_dir.parent
root_dir = test_root_dir.parent
print(root_dir)
print(test_root_dir)
sys.path.append(str(root_dir))

from eddy_squeeze.eddy_squeeze_lib.eddy_files import is_there_any_eddy_output
from eddy_squeeze.eddy_squeeze_lib.eddy_files import get_unique_eddy_prefixes
from eddy_squeeze.eddy_squeeze_lib.eddy_files import EddyRun, EddyRunDPACC
from eddy_squeeze.eddy_squeeze_lib.eddy_files import EddyDirectories

eddy_postfixes = ['post_eddy_shell_alignment_parameters',
                  'outlier_map',
                  'rotated_bvecs',
                  'outlier_n_stdev_map',
                  'outlier_n_sqr_stdev_map',
                  'outlier_report',
                  'parameters',
                  'restricted_movement_rms',
                  'post_eddy_shell_PE_translation_parameters',
                  'values_of_all_input_parameters',
                  'command_txt',
                  'movement_rms']

def test_pass_when_there_is_eddy_output():
    output_dir_files = [
        'user_given_dir/prac.nii.gz'
        'user_given_dir/prac.bval',
        'user_given_dir/prac.bvec',
        'user_given_dir/prac.eddy_command_txt',
        'user_given_dir/prac.eddy_rotated_bvecs',
        'user_given_dir/prac.nii.gz']

    eddy_file_paths = [Path(x) for x in output_dir_files]
    assert is_there_any_eddy_output(eddy_file_paths)

def test_fail_when_there_is_no_eddy_output():
    output_dir_files = [
        'user_given_dir/prac.nii.gz'
        'user_given_dir/prac.bval',
        'user_given_dir/prac.bvec']
    eddy_file_paths = [Path(x) for x in output_dir_files]
    assert is_there_any_eddy_output(eddy_file_paths) == False


def test_fail_when_there_is_no_dir():
    output_dir_files = []
    eddy_file_paths = [Path(x) for x in output_dir_files]
    assert is_there_any_eddy_output(eddy_file_paths) == False


def test_get_unique_eddy_prefixes_when_given_the_same_prefix():
    dot_eddy_files = [
            Path('/data/pnl/prac/haha'),
            Path('/data/pnl/prac/haha')
            ]
    assert get_unique_eddy_prefixes(dot_eddy_files) == dot_eddy_files[0]


def test_get_unique_eddy_prefixes_when_given_one_prefix():
    dot_eddy_files = [
            Path('/data/pnl/prac/haha')
            ]
    assert get_unique_eddy_prefixes(dot_eddy_files) == dot_eddy_files[0]


def test_get_unique_eddy_prefixes_when_given_two_prefix():
    dot_eddy_files = [
            Path('/data/pnl/prac/haha'),
            Path('/data/pnl/prac/hoho')
            ]
    assert get_unique_eddy_prefixes(dot_eddy_files) in dot_eddy_files


def test_get_unique_eddy_prefixes_when_given_no_prefix():
    dot_eddy_files = [
            ]
    with pytest.raises(SystemExit):
        get_unique_eddy_prefixes(dot_eddy_files)


@pytest.fixture
def eddyRunFake():
    eddy_prefix = test_root_dir / 'prac_out' / 'subject02-eddy_out'
    eddyRun = EddyRun(eddy_prefix)
    return eddyRun

def test_EddyRun_fake(eddyRunFake):
    prefix = test_root_dir / 'prac_out'/ 'subject01-eddy_out'
    assert eddyRunFake.eddy_exist is True

@pytest.fixture
def eddyRun():
    eddy_prefix = test_root_dir / 'prac_out' / 'subject01-eddy_out'
    eddyRun = EddyRun(eddy_prefix)
    eddyRun.read_file_locations_from_command()
    eddyRun.load_eddy_information()

    return eddyRun

def test_EddyRun_creation(eddyRun):
    prefix = test_root_dir / 'prac_out'/ 'subject01-eddy_out'
    assert eddyRun.eddy_prefix == prefix
    assert eddyRun.eddy_dir == prefix.parent
    assert eddyRun.eddy_exist is True
    assert eddyRun.subject_name == 'prac_out'

    assert eddyRun.outlier_map == prefix.parent / \
            'subject01-eddy_out.eddy_outlier_map'
    assert eddyRun.outlier_report == prefix.parent / \
            'subject01-eddy_out.eddy_outlier_report'
    assert eddyRun.command_txt == prefix.parent / \
            'subject01-eddy_out.eddy_command_txt'
    assert eddyRun.movement_rms == prefix.parent / \
            'subject01-eddy_out.eddy_movement_rms'
    assert eddyRun.restricted_movement_rms == prefix.parent / \
            'subject01-eddy_out.eddy_restricted_movement_rms'
    assert eddyRun.eddy_out_data == prefix.parent / \
            'subject01-eddy_out.nii.gz'
    assert eddyRun.outlier_free_data == prefix.parent / \
            'subject01-eddy_out.eddy_outlier_free_data.nii.gz'


def test_EddyRun_read_file_locations_from_command(eddyRun):
    # estimate QC values
    assert eddyRun.nifti_input == eddyRun.eddy_prefix.parent / \
            'subject01-dwi-B3000-xc.nii.gz'
    assert eddyRun.bvalue_txt == eddyRun.eddy_prefix.parent / \
            'subject01-dwi-B3000-xc.bval'
    assert eddyRun.mask == eddyRun.eddy_prefix.parent / \
            'subject01-tensormask.nii.gz'

def test_EddyRun_load_eddy_information(eddyRun):

    assert eddyRun.volume_in_each_bshell == {0:5, 200:3, 500:6,
            1000:30, 2950:1, 3000:29}

    # motion array
    movement_array = np.loadtxt(
        eddyRun.eddy_prefix.parent / 'subject01-eddy_out.eddy_movement_rms')
    assert np.equal(eddyRun.movement_array, movement_array).all()

    rms_movement_array = np.loadtxt(
        eddyRun.eddy_prefix.parent / \
            'subject01-eddy_out.eddy_restricted_movement_rms')
    assert np.equal(eddyRun.restricted_movement_array,
                    rms_movement_array).all()

    # outlier arrays
    outlier_std_arr = np.loadtxt(
            eddyRun.eddy_prefix.parent / \
                'subject01-eddy_out.eddy_outlier_n_stdev_map', skiprows=1)
    assert eddyRun.outlier_std_array.mean() == outlier_std_arr.mean()


def test_EddyRun_get_outlier_info(eddyRun):
    eddyRun.get_outlier_info()

def test_EddyRun_summary_df(eddyRun):
    eddyRun.get_outlier_info()
    eddyRun.estimate_eddy_information()
    eddyRun.outlier_summary_df()


def test_EddyDirectories():
    eddy_dirs = [
            (test_root_dir / 'prac_study_dir' / 'subject01'),
            (test_root_dir / 'prac_study_dir' / 'subject02'),
            ]
    # eddyDirectories = EddyDirectories([str(x) for x in eddy_dirs])
    eddyDirectories = EddyDirectories(eddy_dirs)


def test_EddyDirectories_one_dir_with_no_eddy():
    eddy_dirs = [
            (test_root_dir / 'prac_study_dir' / 'subject01'),
            (test_root_dir / 'prac_study_dir' / 'subject02'),
            (test_root_dir / 'prac_study_dir' / 'subject03'),
            ]
    # eddyDirectories = EddyDirectories([str(x) for x in eddy_dirs])
    eddyDirectories = EddyDirectories(eddy_dirs)

def test_EddyDirectories_one_dir_with_no_eddy():
    eddy_dirs = [
            (test_root_dir / 'prac_study_dir' / 'subject01'),
            (test_root_dir / 'prac_study_dir' / 'subject02'),
            (test_root_dir / 'prac_study_dir' / 'subject03'),
            ]
    # eddyDirectories = EddyDirectories([str(x) for x in eddy_dirs])
    eddyDirectories = EddyDirectories(eddy_dirs)
    eddyDirectories.save_all_outlier_slices()


def test_EddyDirectories_nda_upload():
    eddy_dirs = []
    mri_root = Path('/data/predict1/data_from_nda/MRI_ROOT')
    deriv_root = mri_root / 'derivatives'
    dwi_root = deriv_root / 'dwipreproc'
    for ses_dir in dwi_root.glob('sub*/ses*'):
        if (ses_dir / 'eddy_out.nii.gz').is_file():
            eddy_dirs.append(ses_dir)
    # eddy_dirs = [
            # ('/derivatives/dwipreproc/sub-MT13133/ses-202305041'),
            # ]
    eddyDirectories = EddyDirectories(eddy_dirs)
    eddyRun = eddyDirectories.eddyRuns[0]
    print(eddyDirectories.eddyRuns)
    print(eddyDirectories.df_motion.to_csv('motion_summary'))

    print(eddyRun.outlier_summary_df)




def test_one_eddyRun():
    dwi_root = Path('/data/predict1/data_from_nda/MRI_ROOT/derivatives/dwipreproc')
    eddy_prefixes = list(dwi_root.glob('sub-*/ses-*/eddy_out.nii.gz'))
    eddy_prefixes = [x.parent / 'eddy_out' for x in eddy_prefixes]

    df = pd.DataFrame()
    n = 0
    for eddy_prefix in eddy_prefixes:
    # eddy_prefix = dwi_root / 'sub-GA11092/ses-202306021/eddy_out'
    # eddy
    # eddyOut = EddyRun(eddy_prefix, name='ha')
        eddyOut = EddyRunDPACC(eddy_prefix, name='ha')
        eddyOut.bvalue_arr = np.loadtxt(str(eddyOut.bvalue_txt))

        print(eddyOut.bvalue_arr)
        # print(eddyOut.bvalue_txt)
        # eddyOut.load_outlier_arrays()

        try:
            eddyOut.load_eddy_information()
        except OSError:
            continue
        eddyOut.get_info_movement_arrays()

        try:
            eddyOut.estimate_eddy_information()
        except IndexError:
            continue

        print(eddyOut.df)
        df = pd.concat([df, pd.DataFrame(eddyOut.df).T])
        n += 1
        if n == 3:
            pass
            # break


    # print(eddyOut.absolute_movement_array)
    # print(eddyOut.relative_movement_array)
    pass
    df.to_csv('eddy_out_summary.csv')
