from os.path import dirname, join
# from kcho_utils import *
import pandas as pd
from pathlib import Path
import numpy as np
import sys
import nibabel as nb
import re
from tabulate import tabulate
import matplotlib.pyplot as plt

# plot
import seaborn as sns

# warning
import warnings


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

        # The numbers denote how many standard deviations off the mean
        # difference between observation and prediction is.
        self.outlier_std_array = np.loadtxt(self.outlier_n_stdev_map,
                                            skiprows=1)

        # The numbers denote how many standard deviations off the square root
        # of the mean squared difference between observation and prediction is.
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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.outlier_std_mean = self.outlier_std_array[
                np.where(self.outlier_array == 1)].mean()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.outlier_std_std = self.outlier_std_array[
                np.where(self.outlier_array == 1)].std()

        self.outlier_std_mean = np.absolute(self.outlier_std_mean)
        self.outlier_std_std = np.absolute(self.outlier_std_std)

    def read_and_register_raw_files(self):
        """Read eddy command.txt file to load raw data information"""

        # If there is command_txt, get raw dwi input the the eddy from command_txt
        if Path(self.command_txt).is_file():
            with open(self.command_txt, 'r') as f:
                self.command = f.read()

            self.nifti_input = re.search(r'--imain=(\S+)',
                                         self.command).group(1)
            self.bvalue_txt = re.search(r'--bvals=(\S+)',
                                        self.command).group(1)
            self.mask = re.search(r'--mask=(\S+)',
                                  self.command).group(1)

            # quick fix for lupus project, Friday, August 09, 2019
            if '.nii.nii.gz' in self.mask:
                self.mask = re.sub('.nii.nii.gz', '.nii.gz', self.mask)
            elif '.nii.gz' not in self.mask:
                self.mask = self.mask + '.nii.gz'

            # if the file paths were saved as a relative path in the command
            # text file, store them as the absolute path
            for file_name in ['nifti_input', 'bvalue_txt', 'mask']:
                path_in_command = getattr(self, file_name)
                if Path(path_in_command).is_absolute():
                    pass
                else:
                    path_in_command = str(
                        Path(self.ep).parent / Path(path_in_command).name)
                    setattr(self, file_name, path_in_command)

        else:
            sys.exit(f'{self.command_txt} is missing.')

        self.bvalue_arr = np.loadtxt(self.bvalue_txt)


class EddyRun(EddyOut):
    '''Class for FSL eddy output directory'''
    def __init__(self, ep, **kwargs):
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

        # try:
        self.read_and_register_raw_files()
        # except:
            # # TODO : edit here to give options to readin file paths
            # #        when there is no command_txt
            # self.nifti_input = str(self.eddy_dir / 'dti_0107_base.nii.gz')
            # self.bvalue_txt = self.eddy_dir / 'dti_0107_base.bval'
            # self.bvalue_arr = np.loadtxt(self.bvalue_txt)
            # self.mask = str(self.eddy_dir / 'mask_hifi_mask.nii.gz')

        # return number of volume for each shell

        self.volume_in_each_bshell = {}
        shell, count = np.unique(self.bvalue_arr, return_counts=True)
        for shell, count in zip(shell, count):
            self.volume_in_each_bshell[shell] = count

        self.load_movement_arrays()
        self.load_outlier_arrays()
        self.get_info_movement_arrays()
        self.get_info_outlier_arrays()
        self.get_info_post_eddy()
        # print(tabulate(self.post_eddy_shell_alignment_df, headers='keys', tablefmt='psql'))
        # print(tabulate(self.post_eddy_shell_PE_translation_parameters_df, headers='keys', tablefmt='psql'))
        self.collect_all_info()

        if 'name' in kwargs:
            self.df['name_set'] = kwargs.get('name')

    def collect_all_info(self):
        df = pd.Series()

        df['ep'] = self.ep
        df['eddy_dir'] = self.eddy_dir
        df['eddy_input'] = self.nifti_input

        # bvalue
        df['number of volumes'] = len(self.bvalue_arr)
        df['max b value'] = self.bvalue_arr.max()
        df['min b value'] = self.bvalue_arr.min()
        df['unique b values'] = np.unique(self.bvalue_arr)
        df['number of b0s'] = len(self.bvalue_arr[self.bvalue_arr == 0])

        # outlier information
        df['number of outlier slices'] = self.number_of_outlier_slices
        df['Sum of standard deviations in outlier slices'] = \
            self.outlier_std_total
        df['Mean of standard deviations in outlier slices'] = \
            self.outlier_std_mean
        df['Standard deviation of standard deviations in outlier slices'] = \
            self.outlier_std_std

        # movement information
        df['absolute restricted movement'] = self.restricted_movement_avg[0]
        df['relative restricted movement'] = self.restricted_movement_avg[1]
        df['absolute movement'] = self.movement_avg[0]
        df['relative movement'] = self.movement_avg[1]

        self.df = df
        self.post_eddy_shell_alignment_df['ep'] = self.ep
        self.post_eddy_shell_PE_translation_parameters_df['ep'] = self.ep


class EddyDirectory(EddyRun):
    def __init__(self, eddy_dir, **kwargs):
        self.ep = get_unique_eddy_prefixes(eddy_dir)
        EddyRun.__init__(self, self.ep, **kwargs)


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

        print(f'Summarizing {len(self.eddy_dirs)} subjects')

        self.study_eddy_runs = []
        self.eddy_dir_error = []
        self.ep_list = []
        self.eddyRuns = []
        for eddy_dir in self.eddy_dirs:
            eddy_dir_ep = get_unique_eddy_prefixes(eddy_dir)
            self.ep_list.append(eddy_dir_ep)
            eddyRun = EddyRun(eddy_dir_ep)
            # eddyRun = eddy_plots.EddyFigure(
                # eddy_dir,
                # eddy_dir / 'outlier_figure')
            # eddyRun.save_all_outlier_slices()
            self.eddyRuns.append(eddyRun)
            # self.df = pd.concat([self.df, eddyRun.df])

        self.df = pd.concat([x.df.to_frame() for x in self.eddyRuns], axis=1).T
        self.post_eddy_shell_alignment_df = pd.concat(
            [x.post_eddy_shell_alignment_df for x in self.eddyRuns],
            axis=0)
        self.post_eddy_shell_PE_translation_parameters_df = pd.concat(
            [x.post_eddy_shell_PE_translation_parameters_df for x
                in self.eddyRuns],
            axis=0)

    def clean_up_data_frame(self):
        if 'name_set' in self.df.columns:
            self.df.index = self.df.name_set
            self.df.index.name = 'subject'
        else:
            self.df.index = self.df.ep.apply(
                lambda x: Path(x).name.split('-eddy_out')[0]).to_list()
            self.df.index.name = 'subject'
        self.df = self.df.reset_index()

        self.post_eddy_shell_alignment_df.index = \
            self.post_eddy_shell_alignment_df.ep.apply(
                lambda x: Path(x).name.split('-eddy_out')[0]).to_list()
        self.post_eddy_shell_alignment_df.index.name = 'subject'
        self.post_eddy_shell_alignment_df = \
            self.post_eddy_shell_alignment_df.reset_index()

        self.post_eddy_shell_PE_translation_parameters_df.index = \
            self.post_eddy_shell_PE_translation_parameters_df.ep.apply(
                lambda x: Path(x).name.split('-eddy_out')[0]).to_list()
        self.post_eddy_shell_PE_translation_parameters_df.index.name = \
                'subject'
        self.post_eddy_shell_PE_translation_parameters_df = \
            self.post_eddy_shell_PE_translation_parameters_df.reset_index()

    def get_basic_diff_info(self):
        self.df.groupby(
            ['number of volumes',
             'max b value',
             'min b value',
             'number of b0s']).count()['subject'].to_frame()

    def get_unique_bvalues(self):
        unique_b_values = np.stack(self.df['unique b values'])
        return np.unique(unique_b_values, axis=0)

    def plot_subjects(self, var, std_outlier=2):

        # width for one number of subjects
        width_per_subject = 0.5

        g = sns.catplot(x='subject', y=var, data=self.df)
        if len(self.df) < 10:
            g.fig.set_size_inches(5, 5)
        else:
            g.fig.set_size_inches(width_per_subject * len(self.df), 5)
            g.ax.set_xticklabels(g.ax.get_xticklabels(), rotation=90)
        g.fig.set_dpi(200)
        g.ax.set_xlabel('Subjects')

        g.fig.tight_layout()
        g.fig.suptitle(f'{var[0].upper()}{var[1:]}')

        threshold = self.df[var].mean() + (self.df[var].std() * std_outlier)
        g.ax.axhline(y=threshold, color='red', alpha=0.4)
        g.ax.text(
            x=len(self.df) - 0.5,
            y=threshold+0.1,
            s=f'mean + std * {std_outlier}',
            ha='right', color='red', alpha=0.9)

        setattr(self, f'plot_{var}', g)
        # g.fig.show()

        # plot outliers only
        df_tmp = self.df[self.df[var] > threshold]
        x_size = len(df_tmp) * 1.2

        if len(df_tmp) > 0:
            g = sns.catplot(x='subject', y=var, data=df_tmp)
            g.fig.set_size_inches(x_size, 5)
            g.fig.set_dpi(200)
            g.ax.set_xticklabels(g.ax.get_xticklabels())
            g.ax.set_xlabel('Subjects')

            g.fig.suptitle(f'Subjects with greater {var[0].lower()}{var[1:]} '
                           'than (mean + 2*std)', y=1.02)
            setattr(self, f'plot_outlier_only_{var}', g)
        # g.fig.show()

    def figure_post_eddy_shell_PE(self):
        for (title, subtitle), table in \
            self.post_eddy_shell_PE_translation_parameters_df.groupby(
                ['title', 'subtitle']):
            self.figure_post_eddy_shell_PE_suptitle(title, subtitle, table)

    def figure_post_eddy_shell(self):
        for (title, subtitle), table in \
            self.post_eddy_shell_alignment_df.groupby(
                ['title', 'subtitle']):
            self.figure_post_eddy_shell_suptitle(title, subtitle, table)

    def figure_post_eddy_shell_suptitle(self, title, subtitle, t):
        shell_infos = t['shell_info'].unique()
        if len(t) > 30:
            fig, axes = plt.subplots(
                ncols=len(shell_infos),
                figsize=(3.2*len(shell_infos), 2*len(shell_infos)),
                dpi=200)
        else:
            fig, axes = plt.subplots(
                ncols=len(shell_infos),
                figsize=(10, 10),
                dpi=200)

        for ax, shell_info in zip(np.ravel(axes), shell_infos):
            t_tmp = t.groupby('shell_info').get_group(shell_info)
            t_tmp = t_tmp.reset_index()

            std = t_tmp['sum'].std()
            mean = t_tmp['sum'].mean()

            ax.plot(np.arange(len(t_tmp)), t_tmp['sum'], 'ro',
                    alpha=0.3, label='z-rot (deg)')
            ax.axhline(y=mean+2*std, color='r', alpha=0.4)
            ax.text(x=len(t_tmp), y=mean+2*std, s='mean + 2 * std',
                    ha='right', va='top', color='r', alpha=0.4)

            outlier_df = t_tmp[t_tmp['sum'] > (mean + 2*std)]
            for num, row in outlier_df.iterrows():
                ax.text(num, row['sum'],
                        row['subject'],
                        ha='center', fontsize=7)
            print(' '.join(outlier_df.subject.tolist()))
            ax.set_title(shell_info)

        try:
            axes[0].set_ylabel('Sum of xyz translations and rotations')
        except:
            axes.set_ylabel('Sum of xyz translations and rotations')
        fig.suptitle(f'{title}\n{subtitle}', y=1)

        setattr(self, f'plot_post_eddy_shell_{subtitle}', fig)
        # fig.show()

    def figure_post_eddy_shell_PE_suptitle(self, title, subtitle, t):
        shell_infos = t['shell_info'].unique()
        if len(t) > 30:
            fig, axes = plt.subplots(
                ncols=len(shell_infos),
                figsize=(3.2*len(shell_infos), 2*len(shell_infos)),
                dpi=200)
        else:
            fig, axes = plt.subplots(
                ncols=len(shell_infos),
                figsize=(10, 10),
                dpi=200)

        for ax, shell_info in zip(np.ravel(axes), shell_infos):
            t_tmp = t.groupby('shell_info').get_group(shell_info)
            t_tmp = t_tmp.reset_index()

            std = t_tmp[0].std()
            mean = t_tmp[0].mean()

            ax.plot(np.arange(len(t_tmp)), t_tmp[0], 'ro',
                    alpha=0.3, label='z-rot (deg)')
            ax.axhline(y=mean+2*std, color='r', alpha=0.4)
            ax.text(x=len(t_tmp), y=mean+2*std, s='mean + 2 * std',
                    ha='right', va='top', color='r', alpha=0.4)

            outlier_df = t_tmp[t_tmp[0] > (mean + 2*std)]
            for num, row in outlier_df.iterrows():
                ax.text(num, row[0],
                        row['subject'],
                        ha='center', fontsize=7)
            print(' '.join(outlier_df.subject.tolist()))
            ax.set_title(shell_info)

        try:
            axes[0].set_ylabel('PE translation in mm')
        except:
            axes.set_ylabel('PE translation in mm')
        fig.suptitle(f'{title}\n{subtitle}', y=1)
        setattr(self, f'plot_post_eddy_shell_PE_{subtitle}', fig)
        # fig.show()

class EddyDirectories(EddyStudy):
    def __init__(self, eddy_dirs, **kwargs):
        self.eddy_dirs = eddy_dirs
        print(f'Summarizing {len(self.eddy_dirs)} subjects')

        self.study_eddy_runs = []
        self.eddy_dir_error = []
        self.ep_list = []
        self.eddyRuns = []
        for eddy_dir in self.eddy_dirs:
            try:
                eddy_dir_ep = get_unique_eddy_prefixes(eddy_dir)
                self.ep_list.append(eddy_dir_ep)

                if 'name' in kwargs:
                    eddyRun = EddyRun(eddy_dir_ep, name=eddy_dir)
                else:
                    eddyRun = EddyRun(eddy_dir_ep)

                self.eddyRuns.append(eddyRun)
            except:
                pass

        self.df = pd.concat([x.df.to_frame() for x in self.eddyRuns], axis=1).T
        self.post_eddy_shell_alignment_df = pd.concat(
            [x.post_eddy_shell_alignment_df for x in self.eddyRuns],
            axis=0)
        self.post_eddy_shell_PE_translation_parameters_df = pd.concat(
            [x.post_eddy_shell_PE_translation_parameters_df for x
                in self.eddyRuns],
            axis=0)


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
        print(eddy_dir)
        print('There are more than two unique eddy prefix')
        print(dot_eddy_prefix_unique)
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

    for axis in ['x', 'y', 'z']:
        for var in ['-tr (mm)', '-rot (deg)']:
            df[axis+var] = df[axis+var].astype(float).apply(np.absolute)

    df['sum'] = df['x-tr (mm)'] + df['y-tr (mm)'] + df['z-tr (mm)'] + \
        df['x-rot (deg)'] + df['y-rot (deg)'] + df['z-rot (deg)']

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
                pe_translation = re.search(r'(\S+\.|\S+)\ mm', line).group(1)
                pe_translation = np.absolute(float(pe_translation))
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


def eddy(echo_spacing, img_in, bvec, bval, mask,
         eddy_out_prefix, repol_on=True):
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

    if repol_on:
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
    else:
        # eddy_command
        eddy_command = '/data/pnl/soft/pnlpipe3/fsl/bin/eddy_openmp \
            --imain={data} \
            --mask={mask} \
            --index={index} \
            --acqp={acqp} \
            --bvecs={bvecs} \
            --bvals={bvals} \
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
