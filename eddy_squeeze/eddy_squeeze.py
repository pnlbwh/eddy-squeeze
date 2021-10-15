#!/usr/bin/env python
from pathlib import Path
import os
import argparse
from tabulate import tabulate
# development
import sys
import argparse

from eddy_squeeze.eddy_squeeze_lib.eddy_files import EddyDirectories
from eddy_squeeze.eddy_squeeze_lib.eddy_utils import print_header, print_table
from eddy_squeeze.eddy_squeeze_lib.eddy_web import create_study_html, create_html

from typing import List



def get_all_eddy_prefix_paths(eddy_prefix_pattern:str) -> List[Path]:
    '''Get eddy prefix paths from eddy prefix pattern'''
    if Path(eddy_prefix_pattern).is_absolute(): # absolute path
        # [1:] removes '/'
        eddy_prefix_list = list(Path('/').glob(eddy_prefix_pattern[1:]))
    else: # relative path
        eddy_prefix_list = list(Path('.').glob(eddy_prefix_pattern))

    return eddy_prefix_list


def get_eddy_prefix_list(args: 'argparse') -> List[Path]:
    '''Return eddy_prefix_list based on either a pattern or paths'''
    if args.eddy_prefix_pattern:
        print_header(f'Finding matching eddy prefixes')
        eddy_prefix_list = get_all_eddy_prefix_paths(args.eddy_prefix_pattern)
    elif args.eddy_directories:
        print_header(f'Setting up eddy directories')
        eddy_prefix_list = args.eddy_directories
    else:
        sys.exit('Please provide either '
                 '"--eddy_prefix" or "--eddy_directories"')

    return eddy_prefix_list


def eddy_squeeze_study(args: 'argparse') -> None:
    '''Run eddy_squeeze on a group of Eddy outputs'''

    # set eddy summary output directory
    if not args.out_dir:
        out_dir = Path(os.getcwd()) / 'eddy_summary'
    else:
        out_dir = Path(args.out_dir)
    
    print_header(f'Output directory : {out_dir}')

    # get eddy directory prefixes
    eddy_prefix_list = get_eddy_prefix_list(args)

    # get eddyDirectories
    # 1. read files from the command_txt
    # 2. load movement and outlier information
    # 3. clean up outlier slice information
    # 4. clean up movement information
    # 5. collects all information into eddyDirectories.df
    print_header('Extracting information from all eddy outputs')
    eddyDirectories = EddyDirectories(eddy_prefix_list, pnl=args.pnl)
    print_header(f'n={len(eddyDirectories.eddyRuns)} eddy outputs detected')


    # print output
    if args.print_table:
        print_table_from_eddyDirectories(eddyDirectories)

    # create figure
    if args.figures:
        print_header(f'Creating summary figures for all eddy outputs')
        eddyDirectories.save_all_outlier_slices(out_dir)
        eddyDirectories.create_group_figures(out_dir)

    if args.save_html:
        print_header(f'Creating html summary')
        eddyDirectories.save_all_html(out_dir)
        create_study_html(eddyDirectories, out_dir=out_dir)


def print_table_from_eddyDirectories(eddyDirectories:EddyDirectories):
    '''Print three separate tables to the shell'''
    print_header('Basic information')
    print_table(eddyDirectories.subdf_basics)

    print_header('Outlier information')
    print_table(eddyDirectories.subdf_outliers)

    print_header('Motion information')
    print_table(eddyDirectories.subdf_motions)
