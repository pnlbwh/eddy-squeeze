import numpy as np
import pandas as pd
import nibabel as nb
import re, warnings


class EddyCollect():
    def __init__(self):
        pass

    def get_outlier_info(self):
        #TODO add bvalue information to the outlier information
        # find a list of outlier slices
        self.outlier_vol, self.outlier_slice = \
            np.where(self.outlier_array == 1)

        # find a list of std in the outlier slices
        self.stds = self.outlier_std_array[
            self.outlier_vol, self.outlier_slice]

        # find a list of sqr of std in the outlier slices
        self.sqr_stds = self.outlier_sqr_std_array[
            self.outlier_vol,
            self.outlier_slice]

        # get bvalues
        self.outlier_bvalues = self.bvalue_arr[self.outlier_vol]

        # get a order of stds
        # get a order of sqr_stds
        self.rank = (-np.absolute(self.sqr_stds)).argsort().argsort()

    def load_data(self):
        # load data
        self.mask_data = nb.load(self.mask).get_fdata()
        self.pre_data = nb.load(self.nifti_input).get_fdata()
        self.post_data = nb.load(self.outlier_free_data).get_fdata()

        # get mid point of data x-axis
        self.mid_point = int(self.post_data.shape[0] / 2)

    def outlier_summary_df(self):
        """Create summary dataframes
        self.df: details of outlier slices
        self.df_motion: average motion parameter
        """
        df = pd.DataFrame()

        df['Volume'] = self.outlier_vol
        df['Slice'] = self.outlier_slice
        df['B value'] = self.outlier_bvalues
        df['Stds'] = self.stds
        df['Sqr_stds'] = self.sqr_stds
        df['rank'] = self.rank
        df['subject'] = self.subject_name

        self.eddy_outlier_df = df.sort_values(by='rank').reset_index().drop(
            'index',
            axis=1)
        self.eddy_outlier_df.index.name = 'Outlier slices'

        df_motion = pd.DataFrame()
        df_motion['restricted_absolute_motion'] = \
            [self.restricted_movement_avg[0]]
        df_motion['restricted_relative_motion'] = \
            [self.restricted_movement_avg[1]]
        for var in [
                'number_of_outlier_slices',
                'outlier_std_total', 'outlier_std_mean', 'outlier_std_std']:
            df_motion[var] = getattr(self, var)
        self.df_motion = df_motion



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


def get_movement_df(study_eddy_runs):
    '''
    Reads values from list of study_eddy_run classes.

    Args:
        study_eddy_runs: list of Eddy_run classes. list.

    Returns:
        movement_df: table of information for eacch Eddy_run classes.
                     Pandas dataframe.
    '''

    # Ofer measure is removed - Wednesday, July 31, 2019
    #'Ofer measure': [x.ofer_movement_avg for x in study_eddy_runs],

    movement_df = pd.DataFrame()

    for er in study_eddy_runs:
        tmp_dict = {
            'eddy_path': [er.eddy_prefix],
            'Absolute Restricted Movement': [er.restricted_movement_avg[0]],
            'Relative Restricted Movement': [er.restricted_movement_avg[1]],
            'Absolute Movement': [er.movement_avg[0]],
            'Relative Movement': [er.movement_avg[1]],
            'Number of outlier slice': [er.number_of_outlier_slices]
        }
        movement_df_tmp = pd.DataFrame(tmp_dict)
        movement_df = pd.concat([movement_df, movement_df_tmp])

    return movement_df


def get_outlier_df(study_eddy_runs, std_threshold=3):
    '''
    Define which eddy runs are outliers

    Args:
        study_eddy_runs: list of Eddy_run classes. list.
        std_threshold: integer, to be multiplied to the
                       standard deviation to select outliers.
                       Default:3

    Returns:
        pandas dataframe of outlier information
    '''
    # Load data from each eddy run into a dataframe
    movement_df = get_movement_df(study_eddy_runs)

    # Sort by
    movement_df = movement_df.sort_values([
        'Relative Restricted Movement',
        'Number of outlier slice'],
        ascending=False)

    outlier_df = movement_df.copy()

    # For each measure (columns in the movement_df)
    for col in [x for x in outlier_df.columns if x != 'eddy_path']:
        avg = outlier_df[col].mean()
        std = outlier_df[col].std()

        # Create an array that marks outliers
        # True marks outlier
        outlier_df.loc[
                outlier_df[col] > avg + std * std_threshold,
                col] = 'outlier'

        # False marks "almost outlier"
        # Select top 3 nearest subjects to the threshold
        outlier_df.loc[
            (outlier_df.loc[outlier_df[col] != 'outlier', col].sort_values(
                ascending=False)[:3].index),
            col] = 'almost outlier'

    # Drop lines with no True or False
    outlier_df = outlier_df.set_index('eddy_path')
    outlier_df = outlier_df[
        outlier_df.isin(['outlier', 'almost outlier']).any(axis=1)]

    outlier_df[~outlier_df.isin(['outlier', 'almost outlier'])] = '-'

    return outlier_df


class EddyOut:

    def load_movement_arrays(self):
        '''Load information from movement file path attributes'''
        self.restricted_movement_array = np.loadtxt(
            self.restricted_movement_rms)

        self.movement_array = np.loadtxt(self.movement_rms)

    def load_outlier_arrays(self):
        '''Load outlier information'''
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
        '''Extract motion summary from the loaded motion arrays'''

        # restricted movement
        self.restricted_movement_avg = estimate_average_motions(
            self.restricted_movement_rms)

        # general movement
        self.movement_avg = estimate_average_motions(self.movement_rms)
        self.absolute_movement_array = self.movement_array[0, :]
        self.relative_movement_array = self.movement_array[1, :]

    def get_info_post_eddy(self):
        '''Post-eddy information into Pandas DataFrame'''
        self.post_eddy_shell_alignment_df = \
            get_post_eddy_shell_alignment_in_df(
                self.post_eddy_shell_alignment_parameters)

        self.post_eddy_shell_PE_translation_parameters_df = \
            get_post_eddy_shell_PE_translation_parameters(
                self.post_eddy_shell_PE_translation_parameters)

    def get_info_outlier_arrays(self):
        '''Extract information from outlier arrays'''

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

    def load_eddy_information(self):
        '''Load eddy information'''
        
        self.volume_in_each_bshell = \
            get_number_of_directions_in_each_shell_dict(self.bvalue_arr)

        self.load_movement_arrays()
        self.load_outlier_arrays()

    def estimate_eddy_information(self):
        self.get_info_movement_arrays()
        self.get_info_outlier_arrays()
        self.get_info_post_eddy()
        # print(tabulate(self.post_eddy_shell_alignment_df, headers='keys', tablefmt='psql'))
        # print(tabulate(self.post_eddy_shell_PE_translation_parameters_df, headers='keys', tablefmt='psql'))
        self.collect_all_info()


    def collect_all_info(self):
        df = pd.Series()

        df['subject'] = self.subject_name

        df['eddy_prefix'] = self.eddy_prefix
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
        self.post_eddy_shell_alignment_df['eddy_prefix'] = self.eddy_prefix
        self.post_eddy_shell_alignment_df['subject'] = self.subject_name
        self.post_eddy_shell_PE_translation_parameters_df['eddy_prefix'] = self.eddy_prefix
        self.post_eddy_shell_PE_translation_parameters_df['subject'] = \
                self.subject_name


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


def get_number_of_directions_in_each_shell_dict(bvalue_arr):
    '''Return number of directions in each shell'''
    volume_in_each_bshell = {}
    shell, count = np.unique(bvalue_arr, return_counts=True)
    for shell, count in zip(shell, count):
        volume_in_each_bshell[shell] = count
    return volume_in_each_bshell


