from pathlib import Path
import os
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

from eddy_squeeze.eddy_squeeze_lib.eddy_files import EddyRun
from eddy_squeeze.eddy_squeeze_lib.eddy_present import plot_pre_post_correction_slice

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

@pytest.fixture
def eddyRun():
    eddy_prefix = test_root_dir / 'prac_out' / 'subject01-eddy_out'
    eddyRun = EddyRun(eddy_prefix)
    eddyRun.read_file_locations_from_command()
    eddyRun.load_eddy_information()
    eddyRun.get_outlier_info()
    eddyRun.estimate_eddy_information()
    eddyRun.outlier_summary_df()

    return eddyRun


def test_plot_pre_post_correction_slice(eddyRun):
    eddyRun.load_data()
    # plot them
    for v, s, std, sqr_std, r in zip(eddyRun.outlier_vol,
                                     eddyRun.outlier_slice,
                                     eddyRun.stds,
                                     eddyRun.sqr_stds,
                                     eddyRun.rank):
        bvalue = eddyRun.bvalue_arr[v]
        pre_data_tmp = eddyRun.pre_data[:, :, s, v]
        pre_data_tmp = np.where(eddyRun.mask_data[:, :, s] == 1,
                                pre_data_tmp, 0)
        post_data_tmp = eddyRun.post_data[:, :, s, v]
        post_data_tmp = np.where(eddyRun.mask_data[:, :, s] == 1,
                                 post_data_tmp, 0)

        sagittal_data = eddyRun.pre_data[eddyRun.mid_point, :, :, v].T
        sagittal_data_fixed = eddyRun.post_data[eddyRun.mid_point, :, :, v].T

        outfile = f'prac_{r:03}_vol_{v}_slice_{s}.png'

        plot_pre_post_correction_slice(
            eddyRun.eddy_dir,
            pre_data_tmp, post_data_tmp,
            sagittal_data, sagittal_data_fixed,
            outfile,
            s, v, bvalue, r,
            std, sqr_std, eddyRun.outlier_std_array,
            eddyRun.restricted_movement_array)
        break

    assert Path(outfile).is_file()
