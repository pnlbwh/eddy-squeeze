import eddy_squeeze
from pathlib import Path
import pytest
import sys
import imp
script_dir = Path(eddy_squeeze.__file__).parent.parent / 'scripts'
eddy_squeeze_exec = imp.load_source('eddy_squeeze_exec',
                                    str(script_dir / 'eddy_squeeze'))

data_dir = script_dir.parent / 'data'
tmp_study_dir = data_dir / 'tmp_study_dir'


def test_args():

    args = eddy_squeeze_exec.parse_args([
        '--eddy_directories', str(tmp_study_dir / 'subject01'),
        '--out_dir', 'test_out',
        '--print_table'
            ])
    print(args)


def test_simple_run_which_should_fail_due_to_no_bval_bvec_available():

    args = eddy_squeeze_exec.parse_args([
        '--eddy_directories', str(tmp_study_dir / 'subject01'),
        '--out_dir', 'test_out',
        '--print_table'
            ])

    with pytest.raises(SystemExit):
        eddy_squeeze_exec.eddy_squeeze_study(args)


def test_simple_run_with_two_subjects_one_with_correct_data():

    args = eddy_squeeze_exec.parse_args([
        '--eddy_directories', 
            str(tmp_study_dir / 'subject01'),
            str(tmp_study_dir / 'subject02'),
        '--out_dir', 'test_out',
        '--print_table'
            ])

    eddy_squeeze_exec.eddy_squeeze_study(args)


def test_simple_run_with_two_subjects_save_html():

    args = eddy_squeeze_exec.parse_args([
        '--eddy_directories', 
            str(tmp_study_dir / 'subject01'),
            str(tmp_study_dir / 'subject02'),
        '--out_dir', 'test_out',
        '--save_html'
            ])

    eddy_squeeze_exec.eddy_squeeze_study(args)


def test_simple_run_with_two_subjects_save_html_with_figure():

    args = eddy_squeeze_exec.parse_args([
        '--eddy_directories', 
            str(tmp_study_dir / 'subject01'),
            str(tmp_study_dir / 'subject02'),
        '--out_dir', 'test_out',
        '--figure',
        '--save_html'
            ])

    eddy_squeeze_exec.eddy_squeeze_study(args)

