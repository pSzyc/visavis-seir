# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
# Available under GPLv3 licence, see README for more details
    
import numpy as np
from pathlib import Path

import sys
root_repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.defaults import PARAMETERS_DEFAULT


def get_velocity_formula(parameters=PARAMETERS_DEFAULT):
    return 1.25 / (parameters['e_subcompartments_count'] / parameters['e_forward_rate'] + 2 / parameters['c_rate'])

def get_expected_maximum_for_defaults(channel_length):
    return 3.13 * np.sqrt(channel_length) + 81.7  

