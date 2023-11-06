from typing import Any, Optional
from pathlib import Path
import random
import subprocess
import string

import termcolor
import json
import shutil

from .simulation_result import SimulationResult


def _random_name(k: int) -> str:
    return "".join(random.choices(string.ascii_uppercase, k=k))


class VisAVisClient:
    """
    Wraps running simulations into python class.
    """

    def __init__(
        self,
        visavis_bin: Path = Path("./vis-a-vis"),
        sim_root: Path = Path("/tmp"),  # where simulation dirs go
    ):
        if isinstance(visavis_bin, str):
            visavis_bin = Path(visavis_bin)

        if isinstance(sim_root, str):
            sim_root = Path(sim_root)

        self._visavis_bin = visavis_bin
        if not self._visavis_bin.is_file():
            self._visavis_bin = self._visavis_bin.with_suffix(".exe")

        assert self._visavis_bin.is_file()

        self._sim_root = sim_root
        self._sim_root.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        parameters_json: Any,  # ...thing that serializes to json
        protocol_file_path: Path,
        dir_name: Optional[str] = None,  # name the dir with results
        clean_up: bool = True,  # remove files?
        verbose: bool = True,  # print information about progress
        images: bool = False,  # save output images
        activity: bool = True,  # save average activity (E+I) in every row
        states: bool = False,  # save full state
        seed: Optional[int] = None,
    ) -> Optional[SimulationResult]:
        if isinstance(protocol_file_path, str):
            protocol_file_path = Path(protocol_file_path)

        if dir_name is None:
            simulation_dir = self._sim_root / f"sim_{_random_name(16)}"
        else:
            simulation_dir = self._sim_root / dir_name
        simulation_dir.mkdir()

        parameters = simulation_dir / "parameters.json"
        with open(parameters, "w") as f:
            json.dump(parameters_json, f)

        assert protocol_file_path.exists()
        protocol_file_src_path = protocol_file_path
        protocol_file_dst_path = simulation_dir / protocol_file_src_path.name
        shutil.copy(
            protocol_file_src_path.absolute(), protocol_file_dst_path.absolute()
        )

        if verbose:
            termcolor.cprint(
                f"Starting simulation {simulation_dir}\n", end="", color="green"
            )

        ret = subprocess.call(
            [
                self._visavis_bin.absolute(),
                parameters.absolute(),
                protocol_file_dst_path.absolute(),
                *(["--images"] if images else []),
                *(["--activity"] if activity else []),
                *(["--seed", f"{seed}"] if seed is not None else []),
            ],
            cwd=simulation_dir,
            stdout=subprocess.DEVNULL,
        )

        if ret:
            if verbose:
                termcolor.cprint(
                    f"Simulation {simulation_dir} failed\n", end="", color="red"
                )
            if clean_up:
                shutil.rmtree(simulation_dir)
            return None
        else:
            if verbose:
                termcolor.cprint(
                    f"Finished simulation {simulation_dir}\n", end="", color="green"
                )

        res = SimulationResult(simulation_dir, has_states=states, has_activity=activity)

        if clean_up:
            shutil.rmtree(simulation_dir)

        return res
