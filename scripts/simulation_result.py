import re

from pathlib import Path

import pandas as pd


class SimulationResult:
    """
    Wraps directory with simulation results into python class.
    """

    def __init__(self, simulation_dir: Path):
        simulation_dir = Path(simulation_dir)
        self._simulation_dir = simulation_dir
        assert self._simulation_dir.is_dir()

        states_parts = []
        for states_part_path in self._simulation_dir.glob("t_*.csv"):
            (n,) = re.findall(r"(-?\d+)\.csv", states_part_path.name)
            seconds = int(n)

            states_part = pd.read_csv(states_part_path)
            states_part["seconds"] = seconds

            states_parts.append(states_part)

        self._states = pd.concat(states_parts)

    @property
    def simulation_dir(self):
        return self._simulation_dir

    @property
    def states(self):
        return self._states
