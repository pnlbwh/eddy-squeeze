from os.path import dirname, join
# from kcho_utils import *
import pandas as pd
from pathlib import Path
import numpy as np
import sys
import nibabel as nb
import re


class EddyOut:
    def load_movement_arrays(self):
        """Load information from movement file path attributes"""
        self.restricted_movement_array = np.loadtxt(
            self.restricted_movement_rms)

        self.movement_array = np.loadtxt(self.movement_rms)

    def load_outlier_arrays(self):
        """Load outlier information"""
        # Outlier volume, slice, and number
        self.outlier_report_df = get_outlier_report(self.outlier_report)

        # outlier map, stdev, and n_sar_stdev
        self.outlier_array = np.loadtxt(self.outlier_map, skiprows=1)
        self.outlier_std_array = np.loadtxt(self.outlier_n_stdev_map,
                                            skiprows=1)
        self.outlier_sqr_std_array = np.loadtxt(self.outlier_n_sqr_stdev_map,
                                                skiprows=1)

    def get_info_movement_arrays(self):
        """Extract motion summary from the loaded motion arrays"""

        # restricted movement
        self.restricted_movement_avg = estimate_average_motions(
            self.restricted_movement_rms)

        # general movement
        self.movement_avg = estimate_average_motions(self.movement_rms)
        self.absolute_movement_array = self.movement_array[0, :]
        self.relative_movement_array = self.movement_array[1, :]

    def get_info_post_eddy(self):
        """Post-eddy information into Pandas DataFrame"""
        self.post_eddy_shell_alignment_df = \
            get_post_eddy_shell_alignment_in_df(
                self.post_eddy_shell_alignment_parameters)

        self.post_eddy_shell_PE_translation_parameters_df = \
            get_post_eddy_shell_PE_translation_parameters(
                self.post_eddy_shell_PE_translation_parameters)

    def get_info_outlier_arrays(self):
        """Extract information from outlier arrays"""

        om_array = np.loadtxt(self.outlier_map, skiprows=1)
        outlier_slice = np.where(om_array == 1)[1]
        outlier_slice_number = len(outlier_slice)
        self.number_of_outlier_slices = outlier_slice_number

        self.outlier_std_total = self.outlier_std_array[
            np.where(self.outlier_array == 1)].sum()
        self.outlier_std_total = np.absolute(self.outlier_std_total)

        self.outlier_std_mean = self.outlier_std_array[
            np.where(self.outlier_array == 1)].mean()
        self.outlier_std_std = self.outlier_std_array[
            np.where(self.outlier_array == 1)].std()

        self.outlier_std_mean = np.absolute(self.outlier_std_mean)
        self.outlier_std_std = np.absolute(self.outlier_std_std)

    def read_and_register_raw_files(self):
        """Read eddy command.txt file to load raw data information
        - TODO
            - check whether the files exist
        """
        # Get raw dwi input the the eddy from command_txt
        with open(self.command_txt, 'r') as f:
            self.command = f.read()
        self.nifti_input = re.search(r'--imain=(\S+)', self.command).group(1)
        self.bvalue_txt = re.search(r'--bvals=(\S+)', self.command).group(1)
        self.mask = re.search(r'--mask=(\S+)', self.command).group(1)
        # quick fix for lupus project, Friday, August 09, 2019
        if '.nii.nii.gz' in self.mask:
            self.mask = re.sub('.nii.nii.gz', '.nii.gz', self.mask)

        if Path(self.bvalue_txt).is_absolute():
            pass
        else:
            # if self.bvalue_txt is a relative path, assume bvalue_txt is one
            # directory above the eddy output
            self.bvalue_txt = str(
                Path(self.ep).parent / Path(self.bvalue_txt).name)

        self.bvalue_arr = np.loadtxt(self.bvalue_txt)


class EddyRun(EddyOut):
    '''Class for FSL eddy output directory'''
    def __init__(self, ep):
        """EddyRun initialization

        Key Arguments:
            - ep: str, FSL eddy output prefix
                  eg. '/eddy/out/dir/*eddy_out'
        """
        self.ep = ep
        self.eddy_dir = Path(ep).absolute().parent

        # register files
        eddy_files_dict = get_eddy_files(self.ep)
        for name, file_loc in eddy_files_dict.items():
            setattr(self, name, file_loc)

        self.read_and_register_raw_files()
        self.load_movement_arrays()
        self.load_outlier_arrays()
        self.get_info_movement_arrays()
        self.get_info_outlier_arrays()
        self.get_info_post_eddy()


class EddyDirectory(EddyRun):
    def __init__(self, eddy_dir):
        self.ep = get_unique_eddy_prefixes(eddy_dir)
        EddyRun.__init__(self, self.ep)


class EddyStudy:
    '''
    Args:
        study_dir: str, glob input like patterns for eddy directories
        eg) /data/pnl/kcho/*eddy
    '''
    def __init__(self, glob_pattern):
        if Path(glob_pattern).is_absolute():
            self.eddy_dirs = list(Path('/').glob(glob_pattern[1:]))
        else:
            self.eddy_dirs = list(Path('.').glob(glob_pattern[1:]))

        self.study_eddy_runs = []
        self.eddy_dir_error = []
        self.ep_list = []
        for eddy_dir in self.eddy_dirs:
            ep_list = get_unique_eddy_prefixes(eddy_dir)
            for ep in ep_list:
                self.ep_list.append(ep)
                try:
                    self.study_eddy_runs.append(EddyRun(ep))
                except:
                    self.eddy_dir_error.append(ep)


def get_unique_eddy_prefixes(eddy_dir):
    '''
    Returns list of unique eddy prefixes

    Args:
        eddy_dir: string or Path object of eddy out directory
    '''

    dot_eddy_files = [x for x in list(Path(eddy_dir).glob('*.eddy*'))
                      if Path(x).is_file()]

    # ignore QC directory
    dot_eddy_prefix_list = [str(x).split('.eddy')[0] for x in
                            dot_eddy_files
                            if not str(x).endswith('qc')]
    dot_eddy_prefix_unique = list(set(dot_eddy_prefix_list))

    if len(dot_eddy_prefix_unique) == 1:
        return dot_eddy_prefix_unique[0]
    elif len(dot_eddy_prefix_unique) > 1:
        return dot_eddy_prefix_unique[0]
    else:
        sys.exit(f'There is no eddy related files in {eddy_dir}')


def get_eddy_files(ep):
    '''
    Returns dictionary of eddy output file locations in string

    Args:
        ep: string of eddy prefix

    '''
    eddy_6_outputs = ['post_eddy_shell_alignment_parameters',
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

    eddy_files_dict = {}
    for eddy_output in eddy_6_outputs:
        eddy_files_dict[eddy_output] = ep + '.eddy_' + eddy_output

    eddy_files_dict['eddy_out_data'] = ep + '.nii.gz'
    eddy_files_dict['outlier_free_data'] = \
        ep + '.eddy_outlier_free_data.nii.gz'

    return eddy_files_dict


def estimate_average_motions(movement_rms):
    '''
    Return a tuple of averages (absolute, relative)

    Args:
        movement_rms: string of ep.eddy_movement_rms file
    '''
    mean_motion = np.loadtxt(movement_rms)
    motion_avg = mean_motion.mean(axis=0)
    return motion_avg


def make_df_from_lines_post_eddy_shell_alignment(lines):

    df = pd.DataFrame()

    # Get title and subtitle
    for line_number, line in enumerate(lines):
        if re.search('are|were', line):
            title = line
            if lines[line_number+1].startswith('Shell'):
                subtitle = line
        elif len(re.search('[A-Za-z ]*', line).group(0)) > 10:
            subtitle = line
        elif line.startswith('Shell'):
            shell_info = line
            array_info = lines[line_number+2]
            df_tmp = pd.DataFrame(array_info.split()).T
            df_tmp['shell_info'] = shell_info
            df_tmp['title'] = title
            df_tmp['subtitle'] = subtitle

            df = pd.concat([df, df_tmp])

    df.columns = ['x-tr (mm)', 'y-tr (mm)', 'z-tr (mm)',
                  'x-rot (deg)', 'y-rot (deg)', 'z-rot (deg)',
                  'shell_info',
                  'subtitle',
                  'title']
    return df


def get_post_eddy_shell_alignment_in_df(
        post_eddy_shell_alignment_parameters):
    '''
    Return pandas data frame of post eddy shell alignment
    '''

    with open(post_eddy_shell_alignment_parameters, 'r') as f:
        text = f.read()

    lines = text.split('\n')

    df = make_df_from_lines_post_eddy_shell_alignment(lines)

    return df


def make_df_from_lines_post_eddy_shell_PE_translation_parameters(lines):
    df = pd.DataFrame()

    # Get title and subtitle
    for line_number, line in enumerate(lines):
        if re.search('are|were', line):
            title = line
            if lines[line_number+1].startswith('Shell'):
                subtitle = line
        elif re.search(r'PE-translations \(mm\)', line):
            subtitle = line
        elif line.startswith('Shell'):
            shell_info = line.split(':')[0]
            try:
                pe_translation = re.search(r'(\S+\.|)\S+\ mm', line).group(0)
            except:
                pe_translation = ''
            df_tmp = pd.DataFrame([pe_translation])
            df_tmp['shell_info'] = shell_info
            df_tmp['title'] = title
            df_tmp['subtitle'] = subtitle

            df = pd.concat([df, df_tmp])

    return df


def get_post_eddy_shell_PE_translation_parameters(
        post_eddy_shell_PE_translation_parameters):
    '''
    Return pandas data frame of post eddy shell alignment
    '''
    with open(post_eddy_shell_PE_translation_parameters, 'r') as f:
        text = f.read()

    lines = text.split('\n')
    df = make_df_from_lines_post_eddy_shell_PE_translation_parameters(lines)

    return df


def get_outlier_report(outlier_report):
    '''
    Return pandas data frame of eddy outlier report
    '''
    with open(outlier_report, 'r') as f:
        text = f.read()

    df = pd.DataFrame()
    for i in text.split('\n'):
        try:
            slice_number = re.search(r'Slice (\d+)', i).group(1)
            volume_number = re.search(r'scan (\d+)', i).group(1)
            mean_std = re.search(r'mean (\S*\d+.\d+)', i).group(1)
            mean_squared_std = re.search(
                r'mean squared (\S*\d+.\d+)', i).group(1)

            df_tmp = pd.DataFrame(
                    [slice_number,
                     volume_number,
                     mean_std,
                     mean_squared_std]).T
            df = pd.concat([df, df_tmp])
        except:
            #last line has no info
            pass

    try:
        df.columns = ['slice_num', 'vol_num', 'mean_std', 'mean_squared_std']
    except:
        pass

    return df


def eddy(echo_spacing, img_in, bvec, bval, mask, eddy_out_prefix):
    """
    Run FSL eddy

    Parameters:
    echo_spacing (float) : echo spacing of the diffusion data
    other inputs (string)

    https://www.jiscmail.ac.uk/cgi-bin/webadmin?A2=fsl;92bd6f89.1403
        The echo spacing in the Siemens PDF does need to be divided
        by the ipat/GRAPPA factor.  Multi-band has no effect on
        echo spacing.

    https://www.jiscmail.ac.uk/cgi-bin/webadmin?A2=fsl;92136ade.1506
        What I can say is that in _most_ cases it doesn’t really matter
        what you put in the last column. The value is essentially used
        internally to calculate "observed_distortions->estimated_field
        ->estimated_distortions” where the value is used at both -> .
        That means that if you get it wrong, the two errors will cancel.

        It is only really important if
        1. You want your estimated fields to be correctly scaled in Hz
        2. You have acquisitions with different readout times in your
           data set

        Otherwise you can typically just go with 0.05.
        And also make sure that you use the same values for topup and
        eddy if you are using them together.
        Jesper

    https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy/Faq
        There are some special cases where it matters to get the --acqp
        file right, but unless you know exactly what you are doing it
        is generally best to avoid those cases.
        They would be
        - If you acquire data with PE direction along two different
          axes (i.e. the x- and y-axis). In that case you need to get
          the signs right for the columns indicating the PE. But you can
          always use trial and error to find the correct combination.
        - If you acquire data with different readout times. In that case
          you need at the very least to get the ratio between the times
          right.
        - If you use a non-topup derived fieldmap, such as for example
          a dual echo-time gradient echo fieldmap, that you feed into
          eddy as the --field parameter. In this case you need to get
          all signs and times right, both when creating the field (for
          example using prelude) and when specifying its use in eddy
          through the --acqp.

    """
    eddy_out_dir = dirname(eddy_out_prefix)

    # index
    data_img = nb.load(img_in)
    index_array = np.tile(1, data_img.shape[-1])
    index_loc = join(eddy_out_dir, 'index.txt')
    np.savetxt(index_loc, index_array, fmt='%d', newline=' ')

    # acqp
    acqp_num = (128-1) * echo_spacing * 0.001
    acqp_line = '0 -1 0 {}'.format(acqp_num)
    acqp_loc = join(eddy_out_dir, 'acqp.txt')
    with open(acqp_loc, 'w') as f:
        f.write(acqp_line)

    # eddy_command
    eddy_command = '/data/pnl/soft/pnlpipe3/fsl/bin/eddy_openmp \
        --imain={data} \
        --mask={mask} \
        --index={index} \
        --acqp={acqp} \
        --bvecs={bvecs} \
        --bvals={bvals} \
        --repol \
        --out={out}'.format(data=img_in,
                            mask=mask,
                            index=index_loc,
                            acqp=acqp_loc,
                            bvecs=bvec,
                            bvals=bval,
                            out=eddy_out_prefix)

    print(re.sub(r'\s+', ' ', eddy_command))
    run(eddy_command)


def eddy_qc(echo_spacing, bvec, bval, mask, eddy_out_prefix):
    """
    Run FSL eddy qc

    Parameters:
        echo_spacing (float) : echo spacing of the diffusion data
        other inputs (string)

        bvec is not currently used
    """
    eddy_out_dir = dirname(eddy_out_prefix)
    quad_outdir = '{}.qc'.format(eddy_out_prefix)
    index_loc = join(eddy_out_dir, 'index.txt')
    acqp_loc = join(eddy_out_dir, 'acqp.txt')
    command = '/data/pnl/soft/pnlpipe3/fsl/bin/eddy_quad {eddy_out} \
            -idx {index} \
            -par {acqp} \
            -m {nodif_mask} \
            -b {bvals} \
            -o {quad_outdir}'.format(
                eddy_out=eddy_out_prefix,
                index=index_loc,
                acqp=acqp_loc,
                nodif_mask=mask,
                bvals=bval,
                quad_outdir=quad_outdir)

    run(command)

def eddy_squad(qc_dirs, outdir):
    try:
        os.mkdir(outdir)
    except:
        pass
    out_text_file = join(outdir, 'eddy_quad_dirs.txt')
    with open(out_text_file, 'w') as f:
        for qc_dir in qc_dirs:
            f.write('{}\n'.format(qc_dir))

    command = '/data/pnl/soft/pnlpipe3/fsl/bin/eddy_squad \
            {} -o {}'.format(
        out_text_file, outdir)

    run(command)
