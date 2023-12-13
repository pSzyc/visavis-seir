from typing import Iterable, Dict, Callable, Tuple, Iterator
from pathlib import Path
import subprocess
from multiprocessing import Pool
import time


def _with_expanded(fn, kwargs):
    res = fn(**kwargs)
    print('-', end='', flush=True)
    return res


def _with_expanded_kwargs(fn, kwarg_list: Iterable[Dict]) -> Tuple[Callable, Iterator[Dict]]:
    ''' Wraps fn so that 
    multiprocessing.Pool.starmap(*with_expanded_kwargs(fn, kwarg_list))
    is equivalent to
    [fn(**kwargs) for kwargs in kwarg_list]
    '''
    return _with_expanded, ((fn, kwargs) for kwargs in kwarg_list)

def starmap(fn, kwarg_list, processes=None):
    start_time = time.time()
    print('|', end='', flush=True)
    with Pool(processes) as pool:
        result = pool.starmap(*_with_expanded_kwargs(fn, kwarg_list), chunksize=1)
    end_time = time.time()
    elapsed = end_time - start_time
    elapsed_per_it = elapsed / len(kwarg_list)
    print(f'| Took {elapsed:.2f}s  ({elapsed_per_it:.2f}s/it)')
    return result

def simple_starmap(fn, kwarg_list, processes=None):
    start_time = time.time()
    print('|', end='', flush=True)
    result = [fn(**kwargs) for kwargs in kwarg_list]
    end_time = time.time()
    elapsed = end_time - start_time
    elapsed_per_it = elapsed / len(kwarg_list)
    print(f'| Took {elapsed:.2f}s  ({elapsed_per_it:.2f}s/it)')
    return result



def compile_if_not_exists(
    channel_width,
    channel_length,
    path_to_compiling_script=Path(__file__).parent / 'compile_visavis.sh',
    visavis_bin_root=Path(__file__).parent.parent / 'target/bins'
):
    visavis_bin = visavis_bin_root / f'vis-a-vis-{channel_width}-{channel_length}'
    if not Path(visavis_bin).exists() and Path(path_to_compiling_script).exists():
        subprocess.call(
            [str(path_to_compiling_script), '-l', str(channel_length), '-w', str(channel_width)],
            stdout=subprocess.DEVNULL,
            cwd=path_to_compiling_script.parent.parent,
        )
    return visavis_bin

