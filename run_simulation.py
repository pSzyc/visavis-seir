# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
# Available under GPLv3 licence, see README for more details
    
from pathlib import Path
from subplots_from_axsize import subplots_from_axsize
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).parent.parent)) # in order to be able to import from scripts.py

from scripts.plot_result import plot_result, plot_result_from_activity
from scripts.simulation import run_simulation

outdir = Path.cwd() / 'results' 
outdir.mkdir(exist_ok=True, parents=True)

channel_width = 6
channel_length = 200


parameters = {
  "e_subcompartments_count": 4,
  "i_subcompartments_count": 2,
  "r_subcompartments_count": 4,
  "c_rate": 1, # activation rate; = 1 / tau_act
  "e_forward_rate": 1, # rate in min-1. Note that tau_e = e_subcompartments_count / e_forward_rate
  "i_forward_rate": 1, # rate in min-1. Note that tau_e = e_subcompartments_count / e_forward_rate
  "r_forward_rate": 1/15, # rate in min-1. Note that tau_e = e_subcompartments_count / e_forward_rate
} 

result = run_simulation(
    parameters=parameters,
    channel_width=channel_width,
    channel_length=channel_length,
    duration=5, # interval between timepoints (in minutes). Only integer values allowed.
    pulse_intervals=[180] * 6 + [120] * 6 + [60] * 6 +  [900], # list of intervals between pulses (in minutes). Must be int, only multiplicities of "duration" allowed.
    states=True, # dump a csv with the state of the reactor in every timepoint
    activity=True, # dump a csv with the activity averaged across each column of cells
    images=True,  # draw images with the state of the reactor in every timepoint
    clean_up=False, # remove the simulation dir after simulation finished
    sim_root=outdir / 'simulations', # directory to store csvs and images dumped every timepoint. /tmp/qeirq by default
    sim_dir_name=None, # unique subdir of sim_root for the particular simulation. If None, a random name will be used (useful for running multiple instances in parallel)
    save_states=True, # copy all reactor states to a single csv in outdir
    save_activity=True, # copy activity to outdir
    outdir=outdir, # where to store results
    seed=1, # random seed; 0 by default
    verbose=False, 
  )

plot_result_from_activity(result.activity, outfile=outdir / 'kymograph.png', title="6 intervals of 180 min\n6 intervals of 120 min\n6 intervals of 60 min")

