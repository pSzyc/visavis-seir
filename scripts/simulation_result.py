import re

from pathlib import Path

import pandas as pd


class SimulationResult:
    """
    Wraps directory with simulation results into python class.
    """

    def __init__(self, simulation_dir: Path, has_states=True, has_activity=False):
        simulation_dir = Path(simulation_dir)
        self._simulation_dir = simulation_dir
        assert self._simulation_dir.is_dir()


        if has_states:
            self._states = pd.concat(
                pd.read_csv(states_part_path).assign(
                    seconds=int(re.findall(r"state_t(\d+)\.csv", states_part_path.name)[0]),
                    )
                for states_part_path in self._simulation_dir.glob("state_t*.csv")
            )
        else: self._states = None

        if has_activity:
            self._activity = pd.read_csv(simulation_dir / 'activity_column_sum.csv').drop_duplicates('time', keep='last').reset_index(drop=True).rename(columns={'time': 'seconds'})
            self._activity.index.name = 'frame'
            self._activity = self._activity.set_index('seconds', append=True)


    @property
    def simulation_dir(self):
        return self._simulation_dir

    @property
    def states(self):
        return self._states

    @property
    def activity(self):
        return self._activity
