# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
# Available under GPLv3 licence, see README for more details
    
from typing import Any, Optional
from pathlib import Path
import subprocess
from typing import Literal

import termcolor
import json
import shutil

from .simulation_result import SimulationResult
from .utils import random_name


class VisAVisClient:
    """
    Wraps running simulations into a python class.
    """

    def __init__(
        self,
        qeirq_bin: Path = Path("./external/qeirq/target/release/qeirq"),
        path_to_compiling_script = Path(__file__).parent.parent / 'utils' / 'compile_qeirq.sh',
        sim_root: Path = Path("/tmp"),  # where simulation dirs are created
        build: Literal[True, False, 'if needed'] = 'if needed',
    ):

        assert build in [True, False, 'if needed'], build

        if isinstance(qeirq_bin, str):
            qeirq_bin = Path(qeirq_bin)

        if isinstance(sim_root, str):
            sim_root = Path(sim_root)

        self._qeirq_bin = qeirq_bin
        self._path_to_compiling_script = path_to_compiling_script

        if not self._qeirq_bin.is_file() and self._qeirq_bin.with_suffix(".exe").is_file():
            self._qeirq_bin = self._qeirq_bin.with_suffix(".exe")

        if build == True or (build == 'if needed' and not self._qeirq_bin.is_file()):
            self.build()

        if not self._qeirq_bin.is_file() and self._qeirq_bin.with_suffix(".exe").is_file():
            self._qeirq_bin = self._qeirq_bin.with_suffix(".exe")


        assert self._qeirq_bin.is_file()

        self._sim_root = sim_root
        self._sim_root.mkdir(parents=True, exist_ok=True)

    
    def build(self, path_to_compiling_script=None):
        if path_to_compiling_script is None:
            path_to_compiling_script = self._path_to_compiling_script
        if Path(path_to_compiling_script).exists():
            subprocess.call(
                [str(path_to_compiling_script)],
                stdout=subprocess.DEVNULL,
                cwd=path_to_compiling_script.parent.parent,
            )
        else:
            raise FileNotFoundError(f'Script for building qeirq not found at {path_to_compiling_script}')


    def run(
        self,
        parameters_json: Any,  # ...thing that serializes to json
        channel_length: int,
        channel_width: int,
        protocol_file_path: Path,
        dir_name: Optional[str] = None,  # name the dir with results
        clean_up: bool = False,  # remove files?
        verbose: bool = True,  # print information about progress
        images: bool = False,  # save output images
        activity: bool = True,  # save average activity (E+I) in every row
        states: bool = False,  # save full state
        seed: Optional[int] = None,
    ) -> Optional[SimulationResult]:

        if isinstance(protocol_file_path, str):
            protocol_file_path = Path(protocol_file_path)

        if dir_name is None:
            simulation_dir = self._sim_root / f"sim_{random_name(16)}"
        else:
            simulation_dir = self._sim_root / dir_name
        simulation_dir.mkdir()

        parameters_file = simulation_dir / "parameters.json"
        with open(parameters_file, "w") as f:
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
                self._qeirq_bin.absolute(),
                parameters_file.absolute(),
                protocol_file_dst_path.absolute(),
                '--width', str(channel_length),
                '--height', str(channel_width),
                *(["--images-out"] if images else []),
                *(["--states-out"] if states else []),
                *(["--activity-out"] if activity else []),
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
