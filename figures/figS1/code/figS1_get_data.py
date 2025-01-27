# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024-2025 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
 

import pandas as pd
import numpy as np
from pathlib import Path
from subplots_from_axsize import subplots_from_axsize

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent)) # in order to be able to import from scripts.py
from scripts.style import *
from scripts.analyze_velocity import get_velocity

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS1' / 'approach10'
data_dir.mkdir(exist_ok=True, parents=True)

channel_lengths = [30, 300]
channel_widths = list(range(1,10)) + list(range(10,21,2))


velocity, variance_per_step = get_velocities(
    channel_lengths=channel_lengths,
    channel_widths=channel_widths,
    n_simulations=30000,
    outdir=data_dir,
    use_cached=True,
    processes=20,
    return_variance=True,
    )

