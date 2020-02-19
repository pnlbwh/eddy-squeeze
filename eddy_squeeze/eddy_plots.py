#!/usr/bin/env python

from kcho_eddy import EddyDirectory
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib
import sys
import numpy as np
import nibabel as nb
import argparse
import os
import matplotlib.gridspec as gridspec


class EddyFigure(EddyDirectory):
    """Eddy Figure class"""
    def __init__(self, eddy_dir, fig_outdir):
        # initialize with EddyDirectory class
        EddyDirectory.__init__(self, eddy_dir)

        # get detailed information of outlier slices
        self.get_outlier_info()

        # load mask, pre and post-replace data
        self.load_data()

        self.fig_outdir = Path(fig_outdir)
        self.fig_outdir.mkdir(exist_ok=True)

        # get mid point of data x-axis
        self.mid_point = int(self.post_data.shape[0] / 2)

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

    def summary_df(self):
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

        self.df = df.sort_values(by='rank').reset_index().drop(
            'index',
            axis=1)
        self.df.index.name = 'Outlier slices'

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

    def save_all_outlier_slices(self):
        # plot them
        for v, s, std, sqr_std, r in zip(self.outlier_vol,
                                         self.outlier_slice,
                                         self.stds,
                                         self.sqr_stds,
                                         self.rank):
            bvalue = self.bvalue_arr[v]
            pre_data_tmp = self.pre_data[:, :, s, v]
            pre_data_tmp = np.where(self.mask_data[:, :, s] == 1,
                                    pre_data_tmp, 0)
            post_data_tmp = self.post_data[:, :, s, v]
            post_data_tmp = np.where(self.mask_data[:, :, s] == 1,
                                     post_data_tmp, 0)

            sagittal_data = self.pre_data[self.mid_point, :, :, v].T
            sagittal_data_fixed = self.post_data[self.mid_point, :, :, v].T

            outfile = self.fig_outdir / f'{r:03}_vol_{v}_slice_{s}.png'
            plot_pre_post_correction_slice(
                self.eddy_dir,
                pre_data_tmp, post_data_tmp,
                sagittal_data, sagittal_data_fixed,
                outfile,
                s, v, bvalue, r,
                std, sqr_std, self.outlier_std_array,
                self.restricted_movement_array)


def plot_pre_post_correction_slice(
        subject,
        pre_data, post_data, sagittal_data, sagittal_data_fixed, outfile,
        slice_number, volume_number, bvalue, rank,
        outlier_std, outlier_sqr_std,
        std_array, motion_array):
    '''
    Plot pre and post correction slices

    Args:
        pre_data : numpy array of data for the slice
        post_data : numpy array of data for the slice
        out_img : string

    To do:
        set the vmax and vmin equal for both axes
    '''

    fig = plt.figure(constrained_layout=True,
                     figsize=(15, 10))
    gs0 = gridspec.GridSpec(5, 6, figure=fig)

    pre_ax = fig.add_subplot(gs0[0:3, 0:2])
    post_ax = fig.add_subplot(gs0[0:3, 2:4])
    diff_ax = fig.add_subplot(gs0[0:3, 4:6])

    etc_ax = fig.add_subplot(gs0[3, :5])
    sagittal_ax = fig.add_subplot(gs0[3, 5])
    motion_ax = fig.add_subplot(gs0[4, :5])
    sagittal_fixed_ax = fig.add_subplot(gs0[4, 5])
    sagittal_ax.set_axis_off()

    # str matrix
    # graph at the bottom
    all_std_img = etc_ax.imshow(std_array.T,
                                aspect='auto',
                                origin='lower')
    etc_ax.axhline(slice_number, linestyle='--', c='r')
    etc_ax.axvline(volume_number, linestyle='--', c='r')

    etc_ax.set_title(
        'Eddy std array (std of mean difference '
        'between observation and prediction)')
    etc_ax.set_ylabel('Slices')
    etc_ax.set_xticks([])

    # motion
    motion_ax.plot(motion_array[:, 0], label='Absolute')
    motion_ax.plot(motion_array[:, 1], label='Relative')
    motion_ax.axvline(volume_number, linestyle='--', c='r')
    motion_ax.set_title('Restricted motion')
    motion_ax.set_xlabel('Volumes')
    motion_ax.set_ylabel('Motion')
    motion_ax.set_xlim(etc_ax.get_xlim())
    legend = motion_ax.legend(loc='upper left')

    # slice pointer
    sagittal_ax.set_title('Sagittal slice')
    sagittal_ax.imshow(sagittal_data, cmap='gray', origin='lower',
                       aspect='auto')
    val = 3
    sagittal_ax.annotate('',
                         xy=(val, slice_number),
                         xytext=(0, slice_number),
                         arrowprops=dict(facecolor='white',
                                         shrink=0.05),)

    sagittal_ax.annotate('',
                         xy=(sagittal_data.shape[1]-val, slice_number),
                         xytext=(sagittal_data.shape[1], slice_number),
                         arrowprops=dict(facecolor='white',
                                         shrink=0.05),)

    sagittal_fixed_ax.set_title('Sagittal slice (post-eddy)')
    sagittal_fixed_ax.imshow(sagittal_data_fixed,
                             cmap='gray', origin='lower',
                             aspect='auto')
    val = 3
    sagittal_fixed_ax.annotate('',
                               xy=(val, slice_number),
                               xytext=(0, slice_number),
                               arrowprops=dict(facecolor='white',
                                               shrink=0.05),)

    sagittal_fixed_ax.annotate('',
                               xy=(sagittal_data_fixed.shape[1]-val,
                                   slice_number),
                               xytext=(sagittal_data_fixed.shape[1],
                                       slice_number),
                               arrowprops=dict(facecolor='white',
                                               shrink=0.05),)

    post_ax_img = post_ax.imshow(post_data, cmap='gray')
    post_ax.set_title('After Eddy outlier replacement\n'
                      '(before motion correction)')
    post_ax_clim = post_ax_img.get_clim()

    pre_ax_img = pre_ax.imshow(pre_data,
                               cmap='gray',
                               vmin=post_ax_clim[0],
                               vmax=post_ax_clim[1])
    pre_ax.set_title('Before Eddy outlier replacement')

    diff_map = np.sqrt((post_data - pre_data)**2)
    diff_ax_img = diff_ax.imshow(diff_map)

    # colorbar to the diff map
    # [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.95, 0.3746, 0.02, 0.797-0.3746])
    fig.colorbar(diff_ax_img, cax=cbar_ax)
    # diff_ax.set_title('sqrt(diff_map^2)')
    diff_ax.set_title('√(post_data-pre_data)²')

    for ax in post_ax, pre_ax, diff_ax:
        ax.set_axis_off()

    fig.subplots_adjust(left=0.05,
                        right=0.95,
                        bottom=0.07,
                        top=0.80)

    fig.suptitle(
        f'{subject}\n'
        f'Rank by sqr_stds: {rank} Bvalue: {bvalue} Volume {volume_number} Slice {slice_number}\n'
        f'std: {outlier_std:.2f}, sqr_std: {outlier_sqr_std:.2f}',
        y=0.97, fontsize=15)



    #plt.tight_layout()
    fig.savefig(outfile, dpi=fig.dpi)
    plt.close()

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
            'eddy_path': [er.ep],
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


def outlier_df_plot(outlier_df, std_threshold=3):
    '''
    Plot outlier_df

    Args:
        outlier_df: pandas dataframe, containing outlier information.
        std_threshold: integer, to be multiplied to the
                       standard deviation to select outliers.
                       Default:3.

    Returns:
        fig, ax : matplotlib figure and axes.

    To do later:
        - control width according to the size of the input df.
    '''
    map_dict = {'outlier': 1,
                'almost outlier': 0.5,
                '-': 0}
    arr = np.array(outlier_df.applymap(lambda x: map_dict[x]))
    arr = np.where(arr == 0, np.nan, arr)

    fig, ax = plt.subplots(ncols=1, figsize=(20, 10))
    ax.imshow(arr, cmap='Reds', vmax=1, vmin=0)

    cmap = matplotlib.cm.get_cmap('Reds')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

    legend_elements = [
        Patch(facecolor=cmap(norm(1)), label='Outlier'),
        Patch(facecolor=cmap(norm(0.5)), label='Almost outlier')
    ]
    ax.legend(handles=legend_elements,
              loc='center left',
              bbox_to_anchor=(1, 0.5))

    ax.set_yticks(np.arange(len(outlier_df)))
    ax.set_yticklabels(outlier_df.index)

    ax.set_xticks(np.arange(len(outlier_df.columns)))
    ax.set_xticklabels(outlier_df.columns, rotation=90)

    fig.suptitle(
        'Excluded (Threshold : mean + {} std)\n{} subjects'.format(
            std_threshold,
            len(outlier_df)),
        y=.94,
        fontsize=15
    )

    return fig, ax

def motion_summary_figure(study_eddy_runs, std_threshold=3):
    '''
    Draw line graphs of various motion measures from FSL eddy.

    Args:
        study_eddy_runs: list of EddyRun object.
        std_threshold: number to multiply by standard deviation to mark
                       outlier, int.

    Return:
        fig, axes, df
    '''

    movement_df = get_movement_df(study_eddy_runs)
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(20, 20))

    # for each data column in movement_df
    for ax, data, title in zip(
        np.ravel(axes),
        [movement_df[x] for x in movement_df.columns if x != 'eddy_path'],
        [x for x in movement_df.columns if x != 'eddy_path']):

        # line graph for data
        ax.plot(data)

        # Data into array
        arr = np.array(data)

        # Get average motion for all subject
        avg = np.mean(data)

        # Draw an average line
        # and legend to include the number of total subjects
        ax.axhline(y=avg, color='g', alpha=0.8, linestyle='--',
                   label='mean :{}'.format(len(np.array(data))))

        # Get standard deviation for all subject motion
        std = np.std(data)

        # Draw 2*std line
        ax.axhline(y=avg+2*std, color='r', alpha=0.3, linestyle='--',
                   label='mean + 2*std :{}'.format(
                       len(arr[arr>(avg+2*std)])
                   ))

        # Draw 3*std line
        ax.axhline(y=avg+3*std, color='r', alpha=0.9, linestyle='--',
                   label='mean + 3*std :{}'.format(
                       len(arr[arr>(avg+3*std)])
                   ))

        ax.set_title(title)
        ax.legend()

        if title.split(' ')[-1] in ['method', 'relative']:
            ax.set_ylim(0,2)
        ax.set_xlabel('Subject')
        ax.set_ylabel(title)

        if title.startswith('Number'):
            ax.set_ylabel('Number of outlier slice')

    return fig, axes


def shell_alignment_summary(list_of_eddy_runs, std_threshold=3):
    #     list_of_eddy_runs = study_eddy_runs
    df = pd.DataFrame()
    for x in list_of_eddy_runs:
        df_tmp = x.post_eddy_shell_alignment_df
        df_tmp['subject'] = x.ep
        df = pd.concat([df, df_tmp])
        for col in df.columns[:6]:
            df[col] = df[col].astype(float)

    df_long_all =  pd.melt(
            df,
            id_vars=['subtitle', 'title', 'subject', 'shell_info'],
            var_name='motion_parameter',
            value_name='motion')
    df_long_all['axis'] = df_long_all['motion_parameter'].str[0]
    df_long_all['motion_type'] = \
        df_long_all['motion_parameter'].str.extract('(tr|rot)')


    for subtitle_unique, df_for_subtitle in df_long_all.groupby('subtitle'):
        for title_unique, df_for_subtitle_title in df_for_subtitle.groupby('title'):
            # axes
            gb = df_for_subtitle_title.groupby(['shell_info', 'motion_type'])
            ncols = len(df_for_subtitle_title['shell_info'].unique())
            nrows = len(df_for_subtitle_title['motion_type'].unique())
            fig, axes = plt.subplots(
                    ncols=ncols,
                    nrows=nrows,
                    figsize=(ncols*10, nrows*3))

            for ax, (gb_id, df_tmp) in zip(np.ravel(axes, order='F'), gb):
                for axis, table in df_tmp.groupby('axis'):
                    table = table.set_index('subject')
                    table = table.reindex(list(df_long_all.subject.unique()))
                    ax.plot(table['motion'].to_list(), label=axis)

                ax.set_title('\n'.join(gb_id))
                ax.set_xlabel('subject number')
                ax.legend()
            fig.suptitle(title_unique+'\n'+subtitle_unique, y=1.1, fontsize=20)
            fig.tight_layout()

    return fig, axes

def shell_PE_translation_summary(list_of_eddy_runs, std_threshold=3):
    df = pd.DataFrame()
    for x in list_of_eddy_runs:
        df_tmp = x.post_eddy_shell_PE_translation_parameters_df
        df_tmp['subject'] = x.ep
        df = pd.concat([df, df_tmp])

    df[0] = df[0].str.split(' ').str[0].astype(float)
    df.columns = ['motion', 'shell_info', 'title', 'subtitle', 'subject']

    for subtitle_unique, df_for_subtitle in df.groupby('subtitle'):
        for title_unique, df_for_subtitle_title in df_for_subtitle.groupby('title'):
            # axes
            gb = df_for_subtitle_title.groupby('shell_info')
            nrows = len(df_for_subtitle_title['shell_info'].unique())
            fig, axes = plt.subplots(nrows=nrows, figsize=(5*nrows, 10))

            for ax, (gb_id, df_tmp) in zip(np.ravel(axes, order='F'), gb):
                df_tmp = df_tmp.set_index('subject')
                df_tmp = df_tmp.reindex(list(df.subject.unique()))
                ax.plot(df_tmp['motion'].to_list(), label=gb_id)

                ax.set_title(gb_id)
                ax.set_xlabel('subject number')
            fig.suptitle(title_unique+'\n'+subtitle_unique, y=1.1, fontsize=20)
            fig.tight_layout()
            fig.show()

def motion_summary_dist(study_eddy_runs, std_threshold=3):

    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(20, 20))

    outlier_df = pd.DataFrame(columns=['subjects'])

    for ax, data, title in zip(np.ravel(axes), 
                              [[x.ofer_movement_avg for x in study_eddy_runs],
                               [x.movement_avg[1] for x in study_eddy_runs],
                               [x.restricted_movement_avg[1] for x in study_eddy_runs],
                               [x.movement_avg[0] for x in study_eddy_runs],
                               [x.restricted_movement_avg[0] for x in study_eddy_runs],
                               [x.number_of_outlier_slices for x in study_eddy_runs]],
                              ['Ofer motion method', 'Movement : relative', 'Restricted Movement : relative',
                              'Movement : absolute', 'Restricted Movement : absolute', 'Number of outlier slice']):
        ax.hist(data, bins=100)
        avg = np.mean(data)
        std = np.std(data)

        ax.axvline(x=avg, color='g', alpha=0.8, linestyle='--', label='mean')
        ax.axvline(x=avg+2*std, color='r', alpha=0.3, linestyle='--', label='mean + 2*std')
        ax.axvline(x=avg+3*std, color='r', alpha=0.9, linestyle='--', label='mean + 3*std')


        std_threshold = 3
        outlier_data_map = np.where(np.array(data) >  avg+std_threshold*std)
        outlier_eddy_run = list(study_eddy_runs[i] for i in outlier_data_map[0])
        outlier_df_tmp = pd.DataFrame([x.ep for x in outlier_eddy_run], columns=['subjects'])
        outlier_df_tmp[title] = 1

        outlier_df = pd.merge(outlier_df, 
                              outlier_df_tmp,
                              on='subjects', how='outer')

        ax.set_title(title)
        ax.legend()


        ax.set_xlabel(title)
        ax.set_ylabel('Subject count')

        if title.startswith('Number'):
            ax.set_ylabel('Number of outlier slice')

    fig.show()
    outlier_df = outlier_df.fillna(0)


    fig, ax = plt.subplots(ncols=1, figsize=(10, 10))
    ax.imshow(outlier_df.fillna(0).set_index('subjects'))
    ax.set_yticks(np.arange(len(outlier_df)))
    ax.set_yticklabels(outlier_df.subjects)#.str.extract('eddy\/(\d{2}_S\S+)_eddy_corrected')[0])
    ax.set_xticks(np.arange(len(outlier_df.columns[1:])))
    ax.set_xticklabels(outlier_df.columns[1:], rotation=90)
    fig.suptitle('Excluded (Threshold : mean + {} std)\n{} subjects'.format(std_threshold,
                                                                           len(outlier_df)), y=.94, fontsize=15)

    outlier_df['subjects'] = outlier_df.subjects.str.extract('eddy\/(\d{2}_S\S+)_eddy_corrected')
    fig.show()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(prog='Lupus project script',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='''\
    eddy_squeeze -ed eddy/out/dir
''',epilog="Kevin Cho Monday, May 6, 2019")

    argparser.add_argument("--eddy_path_pattern","-ep",
                           type=str,
                           help='specify pattern of the subject'\
                               'eddy folder, '\
                               'eg)\'/data/pnl/projects/NAPLS/DATA/'\
                               'fsleddy_results_2/*/*\' ')

    argparser.add_argument("--eddy_dirs","-ed",
                           type=str,
                           nargs='+',
                           help='specify directories of the eddy folder')
    args = argparser.parse_args()

    argparser.add_argument("--std","-s",
                           type=int,
                           default=3,
                           help='Number multiplied by std to grab outliers')

    argparser.add_argument("--out_dir","-o",
                           type=str,
                           default=os.getcwd(),
                           help='Directory to save outputs')
    args = argparser.parse_args()

    args = argparser.parse_args()

    if args.eddy_path_pattern:
        print('Reading all eddy directories that '\
              'match the pattern : {}'.format(args.eddy_path_pattern))
        eddy_study = kcho_eddy.EddyStudy(args.eddy_path_pattern)
        study_eddy_runs = eddy_study.study_eddy_runs

    elif args.eddy_dirs:
        print('Reading eddy directories')
        study_eddy_runs = []
        for eddy_path in args.eddy_dirs:
            print('\t{}'.format(eddy_path))
            study_eddy_runs.append(kcho_eddy.EddyRun(eddy_path))
    else:
        print("Give --eddy_path_pattern or --eddy_dirs")
        sys.exit()

    # Make output directory if it does not exist
    if not os.path.isdir(args.out_dir):
        os.mkdirs(args.out_dir)

    # Motion graphs
    fig, axes = motion_summary_figure(study_eddy_runs,
                                      args.std)
    fig.savefig('{}/eddy_motion_summary.png'.format(args.out_dir), dpi=fig.dpi)
    plt.close()

    # Raw dataframe
    movement_df = get_movement_df(study_eddy_runs)
    movement_df.to_csv('{}/eddy_movement_and_outlier_slices.csv'.format(args.out_dir))

    # outlier
    outlier_df = get_outlier_df(study_eddy_runs,
                                args.std)
    outlier_df.to_csv('{}/eddy_outlier_marker.csv'.format(args.out_dir))

    fig, ax = outlier_df_plot(outlier_df,
                              args.std)
    plt.tight_layout()
    fig.savefig('{}/eddy_outlier.png'.format(args.out_dir), dpi=fig.dpi)
    plt.close()
    
    print('Done. Summary output directory : {}'.format(args.out_dir))

