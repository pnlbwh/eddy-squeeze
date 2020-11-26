
from pathlib import Path
import os
import sys

root_dir = Path(os.path.realpath(__file__)).parent.parent.parent
sys.path.append(str(root_dir))

from eddy_squeeze.eddy_squeeze_lib.eddy_utils import get_paths_with_suffixes
from eddy_squeeze.eddy_squeeze_lib.eddy_utils import get_absolute_paths
from eddy_squeeze.eddy_squeeze_lib.eddy_utils import get_absolute_when_there_are_dots_in_the_path


def test_attach_nifti_suffixes_if_none():
    output_dir_files = ['user_given_dir/prac',
                        'user_given_dir/prac_mask',
                        'user_given_dir/prac.nii.gz',
                        'user_given_dir/2017.ABC.nii.gz',
                        'user_given_dir/2017.ABC']

    output_dir_paths = [Path(x) for x in output_dir_files]

    corrected_output_paths = get_paths_with_suffixes(output_dir_paths)

    assert corrected_output_paths[0] == \
            output_dir_paths[0].with_suffix('.nii.gz')
    assert corrected_output_paths[1] == \
            output_dir_paths[1].with_suffix('.nii.gz')
    assert corrected_output_paths[2] == output_dir_paths[2]
    assert corrected_output_paths[3] == output_dir_paths[3]
    assert corrected_output_paths[4] == output_dir_paths[4]


def test_get_absolute_when_there_are_dots_in_the_path():
    print()
    root = Path('/data/pnl/kcho/haha')
    output_dir_files = ['/user_given_dir/prac',
                        '../../user_given_dir/prac',
                        '../../../user_given_dir/prac_mask',
                        '../user_given_dir/2017.ABC']
    output_dir_paths = [Path(x) for x in output_dir_files]

    ans_paths = [Path('/user_given_dir/prac'),
                 Path('/data/pnl/user_given_dir/prac'),
                 Path('/data/user_given_dir/prac_mask'),
                 Path('/data/pnl/kcho/user_given_dir/2017.ABC')]

    for path, ans_path in zip(output_dir_paths, ans_paths):
        corr_path = get_absolute_when_there_are_dots_in_the_path(path, root)
        assert corr_path == ans_path


def test_to_absolute_paths_if_relative():
    output_dir_files = ['/root/user_given_dir/prac',
                        'user_given_dir/prac_mask',
                        '../user_given_dir/2017.ABC']

    output_dir_paths = [Path(x) for x in output_dir_files]

    corrected_output_paths = get_absolute_paths(output_dir_paths,
                                                Path('/root'))

    print(corrected_output_paths)
    # assert corrected_output_paths[0] == \
            # output_dir_paths[0].with_suffix('.nii.gz')
    # assert corrected_output_paths[1] == \
            # output_dir_paths[1].with_suffix('.nii.gz')
    # assert corrected_output_paths[2] == output_dir_paths[2]
    # assert corrected_output_paths[3] == output_dir_paths[3]
    # assert corrected_output_paths[4] == output_dir_paths[4]
