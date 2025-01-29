# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024-2025 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
 

from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product

import sys

root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.analyze_periodic import generate_dataset_batch


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig3' / 'fig3C' / 'approach1'
data_dir.mkdir(exist_ok=True, parents=True)


channel_lengths = [300]#, 100, 300, 1000]
channel_widths = [6]#[::-1]
intervals = list(range(30,181,30))#[40,45,50,55,60,65,70,75,80,85,90,95,100,110,120,150,200,300]
n_pulses = 500
n_simulations = 30
ends = [0] 

arrival_times = generate_dataset_batch(
    n_simulations=n_simulations,
    channel_widths=channel_widths,
    channel_lengths=channel_lengths,
    intervals=intervals,
    n_pulses=n_pulses,
    save_iterations=False,
    outdir=data_dir,
    processes=20,
    use_cached=False,
    ends=ends,
    )
    
arrival_times['n_pulses'] = n_pulses
arrival_times['n_simulations'] = n_simulations

arrival_times.reset_index().set_index(['channel_width', 'channel_length', 'interval', 'simulation_id', 'end', 'pulse_id', 'n_simulations', 'n_pulses'])

arrival_times.to_csv(data_dir / (f"arrival_times--l-{'-'.join(map(str,channel_lengths))}.csv"))

