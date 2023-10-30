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


        self._states = pd.concat(
            pd.read_csv(states_part_path).assign(
                seconds=int(re.findall(r"(-?\d+)\.csv", states_part_path.name)[0]),
                )
            for states_part_path in self._simulation_dir.glob("t_*.csv")
        )


    @property
    def simulation_dir(self):
        return self._simulation_dir

    @property
    def states(self):
        return self._states
