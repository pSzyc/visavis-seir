from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import sys
import subprocess
from shutil import rmtree

from .tracking import determine_fates, get_pulse_positions
from .plot_result import plot_result, plot_result_from_activity
from .tracker import make_tracks
from .simulation import run_simulation
from .defaults import TEMP_DIR, PARAMETERS_DEFAULT
from .utils import random_name



def make_video(
    parameters=PARAMETERS_DEFAULT,
    channel_width=6,
    channel_length=300,
    input_protocol=[0],
    duration=5,
    seed=0,
    outdir=None,
    save_video=True,
    snapshots=[],
):

    assert outdir is not None


    sim_dir = Path(TEMP_DIR) / 'qeir' / 'images' / random_name(12)


    result = run_simulation(
        width=channel_width,
        length=channel_length,
        parameters=parameters,
        pulse_intervals=input_protocol,
        seed=seed,
        duration=duration,
        verbose=False,
        states=False,
        activity=True,
        images=True,
        clean_up=False,
        save_states=False,
        save_activity=False,
        sim_root = sim_dir.parent,
        sim_dir_name = sim_dir.name,
        outdir=outdir,
    )

    # input("Press any button to continue...")

    if save_video:
        subprocess.call([
            'ffmpeg', 
            '-framerate', '24',
            '-pattern_type',  'glob',
            '-i', str(sim_dir / 'simulation_results') + '/lattice_*.png',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            # '-vf', f'pad=ceil(iw/8)*8:ceil(ih/8)*8,scale=ceil(iw/8)*2:ceil(ih/8)*2',
            '-vf', f'pad=ceil(iw/8)*8:ceil(ih/8)*8',
            '-preset', 'veryslow',  '-crf', '14',
            # '-vf', f'scale=ceil(iw/4):ceil(ih/4),pad=ceil(iw/8)*2:ceil(ih/8)*2',
            '-y',
            str(outdir / 'video.mp4')
            ])
            # ffmpeg -framerate 24 -pattern_type glob -i '../private/manual/AAHOXEGCFWWE/simulation_results/lattice_00*.png' -c:v libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" out.mp4
    
    rmtree(sim_dir)

    