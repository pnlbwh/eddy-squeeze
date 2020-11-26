from pathlib import Path
import os
import numpy as np
import sys
import pytest

script_loc = Path(os.path.realpath(__file__))
script_dir = script_loc.parent
test_root_dir = script_dir.parent
# root_dir = test_root_dir.parent
print(test_root_dir)
print(test_root_dir)
sys.path.append(str(test_root_dir))

from eddy_squeeze.eddy_squeeze import eddy_squeeze_study
from eddy_squeeze.eddy_squeeze_lib.eddy_files import EddyRun
from eddy_squeeze.eddy_squeeze_lib.eddy_web import create_study_html
from eddy_squeeze.eddy_squeeze_lib.eddy_files import EddyDirectories


class FakeArgs():
    def __init__(self):
        self.out_dir = None
        self.eddy_dir = None
        self.print_only = None
        self.save_csv = None
        self.save_html = None


@pytest.fixture
def eddyRun():
    eddy_prefix = script_dir / 'prac_out' / 'subject01-eddy_out'
    eddyRun = EddyRun(eddy_prefix)
    # eddyRun.read_file_locations_from_command()
    # eddyRun.load_eddy_information()

    return eddyRun


def test_study_create_html(eddyRun):
    fakeArgs = FakeArgs()
    fakeArgs.eddy_prefix = str(eddyRun.eddy_prefix)
    fakeArgs.out_dir = eddyRun.eddy_prefix.parent / 'prac'
    fakeArgs.save_html = True

    eddy_dirs = [
            (script_dir / 'prac_study_dir' / 'subject01'),
            (script_dir / 'prac_study_dir' / 'subject02'),
            (script_dir / 'prac_study_dir' / 'subject03'),
            ]
    # eddyDirectories = EddyDirectories([str(x) for x in eddy_dirs])
    eddyDirectories = EddyDirectories(eddy_dirs)
    create_study_html(eddyDirectories, fakeArgs.out_dir)


def test_eddy_squeeze_study():
    eddy_dirs = [
            (script_dir / 'prac_study_dir' / 'subject01'),
            (script_dir / 'prac_study_dir' / 'subject02'),
            (script_dir / 'prac_study_dir' / 'subject03'),
            ]
    # eddyDirectories = EddyDirectories([str(x) for x in eddy_dirs])
    eddyDirectories = EddyDirectories(eddy_dirs)
    eddyDirectories.create_group_figures(out_dir='eddy_summary')

    # print('Saving the summary outputs to {args.out}')
    # eddyDirectories.df.to_csv('prac_eddy_study_summary.csv')
    create_study_html(eddyDirectories, out_dir='eddy_summary')


def test_eddy_squeeze_study_print_table():
    eddy_dirs = [
            (script_dir / 'prac_study_dir' / 'subject01'),
            (script_dir / 'prac_study_dir' / 'subject02'),
            (script_dir / 'prac_study_dir' / 'subject03'),
            ]
    fakeArgs = FakeArgs()

    fakeArgs.eddy_prefix_pattern = False
    fakeArgs.print_table = True
    fakeArgs.save_html = False
    fakeArgs.eddy_directories = eddy_dirs
    fakeArgs.figures = None

    eddy_squeeze_study(fakeArgs)


def test_eddy_squeeze_study_html():
    eddy_dirs = [
            (script_dir / 'prac_study_dir' / 'subject01'),
            (script_dir / 'prac_study_dir' / 'subject02'),
            (script_dir / 'prac_study_dir' / 'subject03'),
            ]
    fakeArgs = FakeArgs()

    fakeArgs.eddy_prefix_pattern = False
    fakeArgs.print_table = False
    fakeArgs.save_html = True
    fakeArgs.eddy_directories = eddy_dirs
    fakeArgs.figures = None

    eddy_squeeze_study(fakeArgs)

def test_eddy_squeeze_study_html_with_figure():
    eddy_dirs = [
            (script_dir / 'prac_study_dir' / 'subject01'),
            (script_dir / 'prac_study_dir' / 'subject02'),
            (script_dir / 'prac_study_dir' / 'subject03'),
            ]
    fakeArgs = FakeArgs()

    fakeArgs.eddy_prefix_pattern = False
    fakeArgs.print_table = False
    fakeArgs.save_html = True
    fakeArgs.eddy_directories = eddy_dirs
    fakeArgs.figures = True

    eddy_squeeze_study(fakeArgs)


def test_eddy_squeeze_study_html_with_prefix():
    eddy_prefix = str(script_dir / 'prac_study_dir' / 'subj*')
    fakeArgs = FakeArgs()

    fakeArgs.eddy_prefix_pattern = eddy_prefix
    fakeArgs.print_table = False
    fakeArgs.save_html = True
    fakeArgs.eddy_directories = None
    fakeArgs.figures = None

    eddy_squeeze_study(fakeArgs)

