from pathlib import Path
from typing import List
import re, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import seaborn as sns

class EddyFigure(object):
    """Eddy Figure class"""
    def save_all_outlier_slices(self, fig_outdir:str):
        # load mask, pre and post-replace data
        self.load_data()

        self.fig_outdir = Path(fig_outdir)
        self.fig_outdir.mkdir(exist_ok=True, parents=True)

        # plot them
        for v, s, std, sqr_std, r in zip(self.outlier_vol,
                                         self.outlier_slice,
                                         self.stds,
                                         self.sqr_stds,
                                         self.rank):
            outfile = self.fig_outdir / f'{r:03}_vol_{v}_slice_{s}.png'
            if not outfile.is_file():
                bvalue = self.bvalue_arr[v]
                pre_data_tmp = self.pre_data[:, :, s, v]
                pre_data_tmp = np.where(self.mask_data[:, :, s] == 1,
                                        pre_data_tmp, 0)
                post_data_tmp = self.post_data[:, :, s, v]
                post_data_tmp = np.where(self.mask_data[:, :, s] == 1,
                                         post_data_tmp, 0)

                sagittal_data = self.pre_data[self.mid_point, :, :, v].T
                sagittal_data_fixed = self.post_data[self.mid_point, :, :, v].T


                plot_pre_post_correction_slice(
                    self.eddy_dir,
                    pre_data_tmp, post_data_tmp,
                    sagittal_data, sagittal_data_fixed,
                    outfile,
                    s, v, bvalue, r,
                    std, sqr_std, self.outlier_std_array,
                    self.restricted_movement_array)
            else:
                pass


def plot_pre_post_correction_slice(
        subject: Path,
        pre_data: np.array, post_data: np.array,
        sagittal_data: np.array, sagittal_data_fixed: np.array,
        outfile: Path,
        slice_number: float, volume_number: int, bvalue: float, rank: int,
        outlier_std: float, outlier_sqr_std: float,
        std_array: np.array, motion_array: np.array):
    '''Plot pre vsnd post correction slices

    Key Arguments:
        subject : eddy output directory
        pre_data : numpy array of data for the slice, pre-repol
        post_data : numpy array of data for the slice, post-repol
        sagittal_data : numpy array of data for the slice, pre-repol
        sagittal_data_fixed : numpy array of data for the slice, post-repol
        outfile : output png file location, Path
        slice_number : outlier slice number, int
        volume_number: outlier volume number, int
        bvalue: bvalue, float
        rank: rank of the outlier volume, int
        std_array: array of the deviation from gaussian model, np.array
        motion_array: array of the motion (restricted), np.array

    To do:
        set the vmax and vmin equal for both axes
    '''
    # fig = plt.figure(constrained_layout=True,
                     # figsize=(15, 10))
    fig = plt.figure(figsize=(15, 10))
    gs0 = gridspec.GridSpec(5, 6, figure=fig)

    # three graphs at the top
    pre_ax = fig.add_subplot(gs0[0:3, 0:2])
    post_ax = fig.add_subplot(gs0[0:3, 2:4])
    diff_ax = fig.add_subplot(gs0[0:3, 4:6])

    # std and motion graphs
    etc_ax = fig.add_subplot(gs0[3, :5])
    motion_ax = fig.add_subplot(gs0[4, :5])

    # two sagittal slices - to show signal drops
    sagittal_ax = fig.add_subplot(gs0[3, 5])
    sagittal_fixed_ax = fig.add_subplot(gs0[4, 5])
    sagittal_ax.set_axis_off()

    # std graph
    add_eddy_std_array_to_ax(etc_ax, std_array, slice_number, volume_number)

    # motion
    add_motion_graph_to_ax(motion_ax, motion_array, volume_number,
                           etc_ax.get_xlim())

    # two sagital slices
    add_sagital_slices_to_ax(sagittal_ax, sagittal_fixed_ax,
                             sagittal_data,
                             sagittal_data_fixed,
                             slice_number)

    diff_ax_img = add_pre_vs_post_eddy_slices(pre_ax, post_ax, diff_ax,
                                              pre_data, post_data)

    # colorbar to the diff map
    # [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.95, 0.3746, 0.02, 0.797-0.3746])
    fig.colorbar(diff_ax_img, cax=cbar_ax)
    # diff_ax.set_title('sqrt(diff_map^2)')

    fig.subplots_adjust(left=0.05,
                        right=0.95,
                        bottom=0.07,
                        top=0.80)

    fig.suptitle(
        f'{subject}\n'
        f'Rank by sqr_stds: {rank} Bvalue: {bvalue:.0f} ' \
        f'Volume: {volume_number} Slice: {slice_number}\n'
        f'std: {outlier_std:.2f}, sqr_std: {outlier_sqr_std:.2f}',
        y=0.97, fontsize=15)

    #plt.tight_layout()
    fig.savefig(outfile, dpi=fig.dpi)
    plt.close()


def add_eddy_std_array_to_ax(etc_ax:plt.Axes, std_array:np.array,
                             slice_number:int, volume_number:int) -> None:
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


def add_motion_graph_to_ax(motion_ax:plt.Axes, motion_array:np.array,
                           volume_number:int, xlim:np.array) -> None:
    motion_ax.plot(motion_array[:, 0], label='Absolute')
    motion_ax.plot(motion_array[:, 1], label='Relative')
    motion_ax.axvline(volume_number, linestyle='--', c='r')
    motion_ax.set_title('Restricted motion')
    motion_ax.set_xlabel('Volumes')
    motion_ax.set_ylabel('Motion')
    motion_ax.set_xlim(xlim)
    legend = motion_ax.legend(loc='upper left')


def add_sagital_slices_to_ax(sagittal_ax:plt.Axes,
                             sagittal_fixed_ax:plt.Axes,
                             sagittal_data:np.array,
                             sagittal_data_fixed:np.array,
                             slice_number:int) -> None:

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


def add_pre_vs_post_eddy_slices(pre_ax:plt.Axes,
                                post_ax:plt.Axes,
                                diff_ax:plt.Axes,
                                pre_data:np.array,
                                post_data:np.array):
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

    diff_ax.set_title('√(post_data-pre_data)²')

    for ax in post_ax, pre_ax, diff_ax:
        ax.set_axis_off()

    return diff_ax_img


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



class EddyStudyFigures:
    '''
    Args:
        study_dir: str, glob input like patterns for eddy directories
        eg) /data/pnl/kcho/*eddy
    '''
    def set_group_figure_settings(self):
        self.dpi = 100

    def save_all_outlier_slices(self, fig_outdir):
        '''Run all_outlier_slices for all eddyRun objects'''
        for eddyRun in self.eddyRuns:
            subject_fig_outdir = fig_outdir /eddyRun.subject_name
            eddyRun.save_all_outlier_slices(subject_fig_outdir)

    def plot_subjects(self, var, std_outlier=2):
        '''Create graphs for motion, outlier and etc for all eddy outputs'''
        # width for one number of subjects
        width_per_subject = 0.5

        g = sns.catplot(x='subject', y=var, data=self.df)

        if len(self.df) < 10:
            g.fig.set_size_inches(5, 5)
        else:
            g.fig.set_size_inches(width_per_subject * len(self.df), 5)
            g.ax.set_xticklabels(g.ax.get_xticklabels(), rotation=90)

        g.fig.set_dpi(self.dpi)
        g.ax.set_xlabel('Subjects')

        g.fig.tight_layout()
        g.fig.suptitle(f'{var[0].upper()}{var[1:]}')

        threshold = self.df[var].mean() + (self.df[var].std() * std_outlier)
        g.ax.axhline(y=threshold, color='red', alpha=0.4)
        g.ax.text(x=len(self.df) - 0.5,
                  y=threshold+0.1,
                  s=f'mean + std * {std_outlier}',
                  ha='right', color='red', alpha=0.9)

        setattr(self, f'plot_{var}', g)

        # plot outliers only
        df_tmp = self.df[self.df[var] > threshold]
        x_size = len(df_tmp) * 1.2

        if len(df_tmp) > 0:
            g = sns.catplot(x='subject', y=var, data=df_tmp)
            g.fig.set_size_inches(x_size, 5)
            g.fig.set_dpi(self.dpi)
            g.ax.set_xticklabels(g.ax.get_xticklabels())
            g.ax.set_xlabel('Subjects')

            g.fig.suptitle(f'Subjects with greater {var[0].lower()}{var[1:]} '
                           'than (mean + 2*std)', y=1.02)
            setattr(self, f'plot_outlier_only_{var}', g)


    def create_group_figures(self, out_dir):
        '''Create group figures'''
        self.set_group_figure_settings()

        out_dir = Path(out_dir)
        # save figures
        vars = ['absolute restricted movement', 'relative restricted movement',
                'number of outlier slices',
                'Sum of standard deviations in outlier slices',
                'Mean of standard deviations in outlier slices',
                'Standard deviation of standard deviations in outlier slices']
        for var in vars:
            self.plot_subjects(var)
            g = getattr(self, f'plot_{var}')
            g.fig.savefig(out_dir / f'plot_{var}.png')

            # if there is outlier
            try:
                g = getattr(self, f'plot_outlier_only_{var}')
                g.fig.savefig(out_dir / f'plot_outlier_only_{var}.png')
            except:
                pass

        # Eddy shell alignment
        self.figure_post_eddy_shell()
        post_eddy_shell_graph_list = [x for x in dir(self)
                                      if 'plot_post_eddy_shell' in x]
        for post_eddy_shell_graph in post_eddy_shell_graph_list:
            fig = getattr(self, post_eddy_shell_graph)
            fig.savefig(out_dir / f'{post_eddy_shell_graph}.png')


        self.figure_post_eddy_shell_PE()
        post_eddy_shell_PE_graph_list = [x for x in dir(self)
                                         if 'plot_post_eddy_shell_PE' in x]
        for post_eddy_shell_PE_graph in post_eddy_shell_PE_graph_list:
            fig = getattr(self, post_eddy_shell_PE_graph)
            fig.savefig(out_dir / f'{post_eddy_shell_PE_graph}.png')

        # dataframe clean up
        self.df = self.df.sort_values(
            ['number of outlier slices',
             'Sum of standard deviations in outlier slices',
             'absolute restricted movement',
             'relative restricted movement'], ascending=False).drop(
                 ['eddy_prefix', 'eddy_dir', 'eddy_input'],
                 axis=1).reset_index().drop('index', axis=1)

        self.df.to_csv(out_dir / 'eddy_study_summary.csv')


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
                dpi=self.dpi)
        else:
            fig, axes = plt.subplots(
                ncols=len(shell_infos),
                figsize=(10, 10),
                dpi=self.dpi)

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
            # print(' '.join(outlier_df.subject.tolist()))
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
                dpi=self.dpi)
        else:
            fig, axes = plt.subplots(
                ncols=len(shell_infos),
                figsize=(10, 10),
                dpi=self.dpi)

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
            # print(' '.join(outlier_df.subject.tolist()))
            ax.set_title(shell_info)

        try:
            axes[0].set_ylabel('PE translation in mm')
        except:
            axes.set_ylabel('PE translation in mm')
        fig.suptitle(f'{title}\n{subtitle}', y=1)
        setattr(self, f'plot_post_eddy_shell_PE_{subtitle}', fig)
        # fig.show()
