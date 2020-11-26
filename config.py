import os
from pathlib import Path

def get_eddy_location():
    '''Parameters for running eddy-squeeze'''

    config = {}

    # FSL eddy, eddy_quad and eddy_squad paths
    fsldir = Path(os.environ['FSLDIR'])
    config['eddy_location'] = fsldir / 'bin' / 'eddy_openmp'
    config['eddy_quad_location'] = fsldir / 'bin' / 'eddy_quad'
    config['eddy_squad_location'] = fsldir / 'bin' / 'eddy_squad'

    # number of threads to use
    config['ncpu'] = -1

    return config
