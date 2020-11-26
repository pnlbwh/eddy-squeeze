from pathlib import Path
from typing import List
from tabulate import tabulate
import pandas as pd

def print_table(df:pd.DataFrame) -> None:
    '''Prints df using tabulate'''
    # print(tabulate(df, headers='keys', tablefmt='github'))
    print(tabulate(df, headers='keys', tablefmt='psql'))

def get_paths_with_suffixes(paths:list) -> List[Path]:
    '''Return paths with nifti suffix if missing suffix'''

    new_paths = []

    for path in paths:
        suffixes = path.suffixes
        if len(suffixes) == 0:
            new_paths.append(path.with_suffix('.nii.gz'))
        else:
            new_paths.append(path)

    return new_paths


def get_absolute_paths(paths:list, root:Path) -> List[Path]:
    '''Return paths with nifti suffix if missing suffix'''

    new_paths = []

    for path in paths:
        if path.is_absolute():
            new_paths.append(path)
        else:
            if '..' in str(path):
                pass
            else:
                path_absolute = path.absolute()
                new_paths.append(path_absolute)

    return new_paths


def get_absolute_when_there_are_dots_in_the_path(path:Path, root:Path):
    '''Return absolute path when there is dots in the paths'''
    if str(path).startswith('../'):
        new_root = root.parent
        new_path = Path(str(path)[3:])
        absolute_path = get_absolute_when_there_are_dots_in_the_path(
            new_path, new_root)
    else:
        absolute_path = root / path

    return absolute_path


def get_absolute_when_there_are_dots_in_the_paths(paths:Path, root:Path):
    '''Return absolute paths when there is dots in the paths'''
    new_paths = []

    for path in paths:
        new_paths.append(get_absolute_when_there_are_dots_in_the_path(
            path, root))

    return new_paths


def print_header(header:str) -> None:
    '''Print Header'''
    print()
    print(header)
    print('-'*50)

