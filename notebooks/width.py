from track import get_infected_img
from track import get_activations_at_h
import sys
from pathlib import Path
sys.path.insert(0, '..') # in order to be able to import from scripts.py
from scripts.client import VisAVisClient
from scripts.make_protocol import make_protocol

PARAMETERS_DEFAULT = {
  "c_rate": 1,
  "e_incr": 1,
  "i_incr": 1,
  "r_incr": 0.0667
}

intervals = range(100 ,350 ,100)
signal_count = int(sys.argv[1]) # number of signals in each simulations
sim_num = int(sys.argv[2]) # number fo simulations

for chanel_width in range(400, 900, 200):
    folder = f"results/width-{chanel_width}"
    Path(folder).mkdir(parents=True, exist_ok=True)
    client = VisAVisClient(
        visavis_bin=f'../target/bins/vis-a-vis-w-{chanel_width}',
    )

    for interval in intervals:
        protocol_file_path = make_protocol(
            pulse_intervals = (signal_count - 1) * [interval] + [800],
            duration=4,
            out_folder='./'
        )

        for sim in range(sim_num):
            result = client.run(
            parameters_json=PARAMETERS_DEFAULT,
            protocol_file_path= protocol_file_path,
            )
            img = get_infected_img(result.states)
            activations = get_activations_at_h(img, -1)
            activations.to_csv(f"{folder}/out-{interval}-{sim}.csv")