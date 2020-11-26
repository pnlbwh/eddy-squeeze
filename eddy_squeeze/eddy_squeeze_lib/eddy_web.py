from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from pwd import getpwuid
import getpass
from os import stat
import os
import re
import time, datetime
import pandas as pd


root = Path(os.path.abspath(__file__)).parent.parent
static_dir = root.parent / 'docs'
bwh_fig_loc = static_dir / 'pnl-bwh-hms.png'
templates_dir = root / 'html_templates'

# jinja2 environment settings
env = Environment(loader=FileSystemLoader(str(templates_dir)))


# type
from typing import NewType
EddyStudy = NewType('EddyStudy', object)

def basename(path):
    '''functions used in the jinja2 template'''
    return Path(path).name


def sorter(file_path):
    '''functions used in the jinja2 template'''
    return int(file_path.name[:3])


def create_study_html(eddyStudy:EddyStudy, out_dir:str, **kwargs):
    '''Create html that summarizes eddy directorries'''

    # summary out directory settings
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    study_out_html = out_dir / 'eddy_study_summary.html'


    # eddyRun
    env.filters['basename'] = basename
    template = env.get_template('base.html')
    
    html_addresses = []
    for eddyRun in eddyStudy.eddyRuns:
        eddyRun_out_dir = eddyRun.eddy_dir / 'eddy_squeeze_qc'
        image_list = list(sorted(eddyRun_out_dir.glob('*png'), key=sorter))

        out_html = out_dir.absolute() / \
                f'{eddyRun.subject_name}_eddy_summary.html'

        with open(out_html, 'w') as fh:
            fh.write(template.render(image_list=image_list,
                                     eddyOut=eddyRun,
                                     subject=eddyRun.eddy_prefix,
                                     bwh_fig_loc=bwh_fig_loc,
                                     study_out_html=study_out_html
                                     ))

        replace_image_locations_to_relative_in_html(out_html, out_dir)
        html_addresses.append(out_html)

    template = env.get_template('base_study.html')
    # list of images in the output dir created by another function
    image_list = list(out_dir.glob('*png'))
    with open(study_out_html, 'w') as fh:
        fh.write(template.render(out_dir=out_dir,
                                 image_list=image_list,
                                 eddyStudy=eddyStudy,
                                 bwh_fig_loc=bwh_fig_loc,
                                 html_addresses=html_addresses
                                 ))

    replace_image_locations_to_relative_in_html(study_out_html, out_dir)

    from weasyprint import HTML
    HTML(study_out_html).write_pdf(out_dir.absolute() / 'Test.pdf')


def create_html(eddyOut, out_dir:str, **kwargs):
    '''Create html that summarizes individual eddy outputs'''

    if 'out_dir' in kwargs:
        out_dir = Path(kwargs.get('out_dir'))
    else:
        out_dir = eddyOut.eddy_dir / 'outlier_figures'

    out_dir.mkdir(exist_ok=True)
    image_list = list(sorted(out_dir.glob('*png'), key=sorter))

    git_hash = get_git_hash()
    env.filters['basename'] = basename
    template = env.get_template('base.html')

    out_html = out_dir.absolute() / 'eddy_summary.html'
    with open(out_html, 'w') as fh:
        fh.write(template.render(image_list=image_list,
                                 eddyOut=eddyOut,
                                 subject=eddyOut.eddy_prefix,
                                 bwh_fig_loc=bwh_fig_loc,
                                 ))

    replace_image_locations_to_relative_in_html(out_html, out_dir)


def replace_image_locations_to_relative_in_html(
        html_loc:str, image_root: Path) -> None:
    '''Replace image locations to relative location'''
    # Read in the file
    with open(html_loc, 'r') as file :
      filedata = file.readlines()

    # Replace the target string
    new_lines = []
    for line in filedata:
        if 'img src' in line:
            new_line = re.sub(f'{image_root.absolute()}/', '', line)
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    # Write the file out again
    with open(html_loc, 'w') as file:
        for new_line in new_lines:
          file.write(new_line)
    

def get_git_hash() -> str:
    # git version
    command = 'git rev-parse HEAD'
    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_dir)
    git_hash = os.popen(command).read()
    return git_hash
