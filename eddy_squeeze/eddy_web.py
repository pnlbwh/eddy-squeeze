from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from pwd import getpwuid
import getpass
from os import stat
import os
import re
import time, datetime
import pandas as pd


def basename(path):
    return Path(path).name


def sorter(file_path):
    return int(file_path.name[:3])


def create_study_html(eddyStudy, **kwargs):
    """Create html that summarizes eddy directoreis"""
    if 'out_dir' in kwargs:
        out_dir = kwargs.get('out_dir')
    else:
        out_dir = os.getcwd()

    image_list = list(out_dir.glob('*png'))

    root = os.path.dirname(os.path.abspath(__file__))
    static_dir = Path(root).parent / 'docs'
    bwh_fig_loc = static_dir / 'pnl-bwh-hms.png'

    templates_dir = os.path.join(root, 'templates')
    env = Environment(loader=FileSystemLoader(templates_dir))

    template = env.get_template('base_study.html')

    (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = \
        os.stat(Path(out_dir) / 'eddy_study_summary.csv')
    time_now = time.strftime('%Y-%m-%d', time.localtime(mtime))
    print(time_now)

    filename = out_dir / 'eddy_study_summary.html'

    print(eddyStudy.df)
    print(eddyStudy.df.columns)
    with open(filename, 'w') as fh:
        fh.write(template.render(
            out_dir=out_dir,
            image_list=image_list,
            eddyStudy=eddyStudy,
            bwh_fig_loc=bwh_fig_loc
            ))

def create_html(eddyOut, **kwargs):
    """Create html that summarizes randomise_summary.py outputs"""

    if 'out_dir' in kwargs:
        out_dir = kwargs.get('out_dir')
    else:
        out_dir = eddyOut.eddy_dir / 'outlier_figures'

    image_list = list(sorted(out_dir.glob('*png'), key=sorter))

    
    # git version
    command = 'git rev-parse HEAD'
    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_dir)
    git_hash = os.popen(command).read()


    root = os.path.dirname(os.path.abspath(__file__))

    static_dir = Path(root).parent / 'docs'
    bwh_fig_loc = static_dir / 'pnl-bwh-hms.png'

    templates_dir = os.path.join(root, 'templates')
    env = Environment(loader=FileSystemLoader(templates_dir))
    env.filters['basename'] = basename
    template = env.get_template('base.html')

    (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = \
        os.stat(eddyOut.nifti_input)
    time_now = time.strftime('%Y-%m-%d', time.localtime(mtime))
    print(time_now)

    filename = out_dir / 'eddy_summary.html'

    with open(filename, 'w') as fh:
        fh.write(template.render(
            image_list=image_list,
            eddyOut=eddyOut,
            subject=eddyOut.ep,
            bwh_fig_loc=bwh_fig_loc,
            jkhaha='hoho'))
