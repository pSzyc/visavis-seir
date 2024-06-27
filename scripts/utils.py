# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
# Available under GPLv3 licence, see README for more details
    
from typing import Iterable, Dict, Callable, Tuple, Iterator
from pathlib import Path
import subprocess
from multiprocessing import Pool
import time
import string
import random


def random_name(k: int) -> str:
    return "".join(random.choices(string.ascii_uppercase, k=k))


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
        processes = pool._processes
    end_time = time.time()
    elapsed = end_time - start_time
    elapsed_per_it = elapsed / ((len(kwarg_list)-1) // processes + 1)
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

