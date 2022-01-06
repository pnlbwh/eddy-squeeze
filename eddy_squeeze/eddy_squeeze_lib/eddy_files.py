from pathlib import Path
from typing import List
import re, sys
import pandas as pd
import numpy as np

from eddy_squeeze.eddy_squeeze_lib.eddy_collect import EddyOut, EddyCollect
from eddy_squeeze.eddy_squeeze_lib.eddy_present import EddyFigure, EddyStudyFigures
from eddy_squeeze.eddy_squeeze_lib.eddy_utils import get_paths_with_suffixes
from eddy_squeeze.eddy_squeeze_lib.eddy_utils import get_absolute_when_there_are_dots_in_the_paths
from eddy_squeeze.eddy_squeeze_lib.eddy_web import create_html



class EddyRun(EddyOut, EddyCollect, EddyFigure):
    '''EddyRun to catch user given eddy prefix 
       to return object with eddy info'''
    def __init__(self, eddy_prefix:str, **kwargs):
        '''EddyRun initialization

        Key Arguments:
            - eddy_prefix: Eddy output prefix, str
                  eg. '/eddy/out/dir/*eddy_out'

        kwargs arguments:
            - name: name of the data or subject
            - dwi, bval, mask: paths of raw dwi, bval and mask.
                               Giving these paths will overwrite the locatios
                               defined in the eddy command.txt. And will be
                               used to create eddy repol difference figures
        '''

        self.eddy_prefix = Path(eddy_prefix)
        self.eddy_dir = self.eddy_prefix.absolute().parent
        self.eddy_exist = check_if_eddy_ran(self.eddy_dir)

        if 'name' in kwargs:
            self.subject_name = kwargs.get('name')
        else:
            self.subject_name = \
                get_subject_name_from_eddy_prefix(self.eddy_prefix)

        register_eddy_paths(self)

    def read_file_locations_from_command(self, **kwargs):
        '''Register raw inputs to the eddy'''

        # register nifti_input, bval and mask to self
        # based on the file paths in command_txt
        paths = return_paths_from_eddy_command_txt(self.command_txt)
        self.nifti_input, self.bvalue_txt, self.mask = paths


        # this is required for loading pre-repol and post-repol
        # overwrite paths if it's given through kwargs
        if 'dwi' in kwargs:
            self.nifti_input = Path(kwargs.get('dwi'))
        if 'bval' in kwargs:
            self.bvalue_txt = Path(kwargs.get('dwi'))
        if 'mask' in kwargs:
            self.mask = Path(kwargs.get('dwi'))

        self.bvalue_arr = np.loadtxt(str(self.bvalue_txt))

    # def read_and_register_raw_files(self):
        # '''Read eddy command.txt file to load raw data information'''

        # # If there is command_txt, get raw dwi input 
        # # the the eddy from command_txt
        # paths = return_paths_from_eddy_command_txt(self.command_txt)
        # self.nifti_input, self.bvalue_txt, self.mask = paths

        # self.bvalue_arr = np.loadtxt(self.bvalue_txt)


class EddyDirectory(EddyRun):
    def __init__(self, eddy_dir, **kwargs):
        dot_eddy_files = get_dot_eddy_files(eddy_dir)
        self.ep = get_unique_eddy_prefixes(eddy_dir)
        EddyRun.__init__(self, self.ep, **kwargs)


def get_dot_eddy_files(eddy_dir: str) -> List[Path]:
    '''Returns list of paths, that includes .eddy'''
    eddy_files_and_dirs = list(Path(eddy_dir).glob('*.eddy*'))
    dot_eddy_files = [str(x).split('.eddy')[0] for x in eddy_files_and_dirs if Path(x).is_file()]

    if len(dot_eddy_files) == 0:
        print(f'There is no eddy related files in {eddy_dir}')

    return dot_eddy_files


def get_unique_eddy_prefixes(dot_eddy_files: List[Path]) -> str:
    '''
    Returns list of unique eddy prefixes

    Args:
        eddy_dir: string or Path object of eddy out directory
    '''
    dot_eddy_prefix_unique = set(dot_eddy_files)

    if len(dot_eddy_prefix_unique) == 1:
        return dot_eddy_prefix_unique.pop()
    elif len(dot_eddy_prefix_unique) > 1:
        print('There are more than two unique eddy prefix')
        print(f'\t{dot_eddy_prefix_unique}')
        return dot_eddy_prefix_unique.pop()
    else:
        sys.exit(f'There is no eddy related files')


def is_there_any_eddy_output(paths: List[Path]):
    '''Return True if there is any path that matches eddy output'''
    paths_including_eddy_in_fname = [x for x in paths if '.eddy' in x.name]

    if len(paths_including_eddy_in_fname) > 1:
        return True
    else:
        return False


def check_if_eddy_ran(user_given_dir:str):
    '''Return True if there are eddy outputs'''
    user_given_dir = Path(user_given_dir)

    paths_with_eddy_in_name = user_given_dir.glob('*.eddy*')
    is_eddy_ran = is_there_any_eddy_output(paths_with_eddy_in_name)

    return is_eddy_ran


def get_subject_name_from_eddy_prefix(eddy_prefix:Path, pattern=None):
    '''Return subject name from eddy prefix'''

    # to absolute path
    eddy_prefix = eddy_prefix.absolute()

    if pattern is not None:
        subject_name = re.search(pattern, str(eddy_prefix.parent)).group(1)
    else:
        subject_name = eddy_prefix.parent.parent.name

    return subject_name


def get_eddy_files(eddy_prefix) -> dict:
    '''
    Returns dictionary of eddy output file locations in string

    Args:
        EddyRun
        eddy_prefix: string of eddy prefix

    '''
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

    eddy_files_dict = {}
    for eddy_postfix in eddy_postfixes:
        eddy_files_dict[eddy_postfix] = eddy_prefix.with_suffix(
            f'.eddy_{eddy_postfix}')

    eddy_files_dict['eddy_out_data'] = eddy_prefix.with_suffix('.nii.gz')
    eddy_files_dict['outlier_free_data'] = eddy_prefix.with_suffix(
        '.eddy_outlier_free_data.nii.gz')

    return eddy_files_dict


def register_eddy_paths(eddyRun:EddyRun) -> None:
    '''Register eddy out files to attributes

    Key arguments:
        - eddyRun
    '''
    eddy_files_dict = get_eddy_files(eddyRun.eddy_prefix)
    for name, file_loc in eddy_files_dict.items():
        setattr(eddyRun, name, file_loc)


def return_paths_from_eddy_command_txt(command_txt_loc:Path) -> tuple:
    """Read eddy command.txt file to load raw data information"""

    with open(command_txt_loc, 'r') as f:
        command = f.read()

    nifti_input = Path(re.search(r'--imain=(\S+)', command).group(1))
    bvalue_txt = Path(re.search(r'--bvals=(\S+)',command).group(1))
    mask = Path(re.search(r'--mask=(\S+)', command).group(1))

    nifti_input, mask = get_paths_with_suffixes([nifti_input, mask])

    nifti_input, bvalue_txt, mask = \
        get_absolute_when_there_are_dots_in_the_paths(
            [nifti_input, bvalue_txt, mask], command_txt_loc.parent)

    return (nifti_input, bvalue_txt, mask)



class EddyDirectories(EddyStudyFigures):
    def __init__(self, eddy_dirs:List[Path], **kwargs):
        self.eddy_dirs = eddy_dirs
        print(f'Summarizing {len(self.eddy_dirs)} subjects')

        self.study_eddy_runs = []
        self.eddy_dir_error = []
        self.eddy_prefix_list = []
        self.eddyRuns = []
        for eddy_dir in self.eddy_dirs:
            try:
                dot_eddy_files = get_dot_eddy_files(eddy_dir)
                eddy_dir_ep = get_unique_eddy_prefixes(dot_eddy_files)
                self.eddy_prefix_list.append(eddy_dir_ep)

                if 'name' in kwargs:
                    eddyRun = EddyRun(eddy_dir_ep, name=eddy_dir)
                else:
                    eddyRun = EddyRun(eddy_dir_ep)

                eddyRun.subject_name = eddyRun.eddy_dir.parent.name

                try:
                    eddyRun.read_file_locations_from_command()
                except:
                    orig_bvalue_txt = eddyRun.bvalue_txt
                    eddyRun.nifti_input = eddyRun.eddy_dir / \
                            Path(eddyRun.nifti_input).name
                    eddyRun.bvalue_txt = eddyRun.eddy_dir / \
                            Path(eddyRun.bvalue_txt).name
                    eddyRun.mask = eddyRun.eddy_dir / \
                            Path(eddyRun.mask).name

                    if not eddyRun.bvalue_txt.is_file():
                        sys.exit('There is no bval files. The script tried '
                                 f'loading: \n\t-{eddyRun.bvalue_txt}\n'
                                 f'\t-{orig_bvalue_txt}\n')
                    eddyRun.bvalue_arr = np.loadtxt(eddyRun.bvalue_txt)

                # if PNL structure
                if kwargs.get('pnl'):
                    eddyRun.dwi_dir = eddyRun.eddy_dir.parent
                    eddyRun.session_dir = eddyRun.dwi_dir.parent
                    eddyRun.session_name = eddyRun.session_dir.name
                    eddyRun.subject_root = eddyRun.session_dir.parent
                    eddyRun.subject_name = eddyRun.subject_root.name

                    eddyRun.eddy_out_data = eddyRun.eddy_dir.parent / \
                        eddyRun.eddy_out_data.name

                    eddyRun.eddy_out_data = eddyRun.eddy_dir.parent / \
                        f'{eddyRun.subject_name}_{eddyRun.session_name}_desc-XcUnEd_dwi.nii.gz'

                eddyRun.load_eddy_information()
                eddyRun.get_outlier_info()
                eddyRun.estimate_eddy_information()
                eddyRun.outlier_summary_df()
                eddyRun.prepared = True

                self.eddyRuns.append(eddyRun)


            except SystemExit:
                pass
                # pass
        # self.df = pd.concat([x.df.to_frame() for x in self.eddyRuns], axis=1).T

        if all([x.prepared == False for x in self.eddyRuns]):
            sys.exit('Errors in all input error directories')

        self.df = pd.concat([x.df for x in self.eddyRuns], axis=1).T
        self.create_sub_df_for_each_information()

        self.df_motion = pd.concat([x.df_motion for x in self.eddyRuns])

        self.post_eddy_shell_alignment_df = pd.concat(
            [x.post_eddy_shell_alignment_df for x in self.eddyRuns],
            axis=0)
        self.post_eddy_shell_PE_translation_parameters_df = pd.concat(
            [x.post_eddy_shell_PE_translation_parameters_df for x
                in self.eddyRuns],
            axis=0)

    def save_all_html(self, fig_root_dir):
        '''Run all_outlier_slices for all eddyRun objects'''
        for eddyRun in self.eddyRuns:
            fig_outdir = fig_root_dir / eddyRun.subject_name
            create_html(eddyRun, out_dir=fig_outdir)

    def get_unique_bvalues(self):
        unique_b_values = np.stack(self.df['unique b values'])
        return np.unique(unique_b_values, axis=0)

    def create_sub_df_for_each_information(self):
        '''Divide self.df into different dfs for better visualization'''

        self.subdf_basics = self.df[
            ['subject', 'eddy_dir', 'number of volumes', 'max b value',
             'min b value', 'unique b values', 'number of b0s']]

        self.subdf_outliers = self.df[
            ['subject', 'number of outlier slices',
             'Sum of standard deviations in outlier slices',
             'Mean of standard deviations in outlier slices',
             'Standard deviation of standard deviations in outlier slices']]

        self.subdf_motions = self.df[
            ['subject', 'absolute restricted movement',
             'relative restricted movement']]

    def clean_up_data_frame(self):
        pass
        # if 'name_set' in self.df.columns:
            # self.df.index = self.df.name_set
        # else:
            # self.df.index = self.df.ep.apply(
                # lambda x: Path(x).name.split('-eddy_out')[0]).to_list()

        # self.df.index.name = 'subject'
        # self.df = self.df.reset_index()

        # self.post_eddy_shell_alignment_df.index = \
            # self.post_eddy_shell_alignment_df.subject

        # self.post_eddy_shell_alignment_df.index.name = 'subject'
        # self.post_eddy_shell_alignment_df = \
            # self.post_eddy_shell_alignment_df.reset_index()

        # self.post_eddy_shell_PE_translation_parameters_df.index = \
            # self.post_eddy_shell_PE_translation_parameters_df.ep.apply(
                # lambda x: Path(x).name.split('-eddy_out')[0]).to_list()
        # self.post_eddy_shell_PE_translation_parameters_df.index.name = \
                # 'subject'
        # self.post_eddy_shell_PE_translation_parameters_df = \
            # self.post_eddy_shell_PE_translation_parameters_df.reset_index()
