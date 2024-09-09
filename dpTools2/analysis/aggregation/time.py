import numpy as np
import numba as nb
from collections import defaultdict
import multiprocessing as mp
import concurrent.futures
from functools import partial
from tqdm import tqdm
import os
from typing import List, Callable, Any

def calculate_serial(indices: List[List[int]], calc_func: Callable, **func_kwargs) -> List[Any]:
    """
    Calculate the series of values for a given set of indices using a provided calculation function sequentially.

    Parameters
    ----------
    indices : List[List[int]]
        A list of lists of indices, where each sublist represents a set of indices to be processed.
    calc_func : Callable
        The function to be applied to each set of indices.
        Signature must be `calc_func(indices: List[int], **func_kwargs) -> Any`.
    func_kwargs : dict
        Keyword arguments to be passed to the calculation function.

    Returns
    -------
    List[Any]
        A list of results from the calculation function, in the same order as the input indices.
    """
    results = []
    for index_set in tqdm(indices):
        results.append(calc_func(index_set, **func_kwargs))
    return results

def calculate_parallel_threadpool(indices: List[List[int]], calc_func: Callable, chunk_size: int = 1, cpu_count: int = None, **func_kwargs) -> List[Any]:
    """
    Calculate the series of values for a given set of indices using a provided calculation function in parallel.

    Parameters
    ----------
    indices : List[List[int]]
        A list of lists of indices, where each sublist represents a set of indices to be processed.
    calc_func : Callable
        The function to be applied to each set of indices.
        Signature must be `calc_func(indices: List[int], **func_kwargs) -> Any`.
    func_kwargs : dict
        Keyword arguments to be passed to the calculation function.
    chunk_size : int, optional
        The size of the chunks to be processed in parallel (default is 1).
    cpu_count : int, optional
        The number of CPU cores to be used for parallel processing (default is the number of available cores).

    Returns
    -------
    List[Any]
        A list of results from the calculation function, in the same order as the input indices.
    """
    cpu_count = os.cpu_count() if cpu_count is None else cpu_count
    chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count) as executor:
        futures = []
        for chunk in chunks:
            futures.append(executor.submit(
                lambda ch: [calc_func(index_set, **func_kwargs) for index_set in ch], chunk))
        
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            results.extend(future.result())
    return results

def calculate_parallel_multiprocessing(indices: List[List[int]], calc_func: Callable, chunk_size: int = 1000, cpu_count: int = None, **func_kwargs) -> List[Any]:
    """
    Calculate the series of values for a given set of indices using a provided calculation function in parallel.

    Parameters
    ----------
    indices : List[List[int]]
        A list of lists of indices, where each sublist represents a set of indices to be processed.
    calc_func : Callable
        The function to be applied to each set of indices.
        Signature must be `calc_func(indices: List[int], **func_kwargs) -> Any`.
    cpu_count : int, optional
        The number of CPU cores to be used for parallel processing (default is the number of available cores).

    Returns
    -------
    List[Any]
        A list of results from the calculation function, in the same order as the input indices.
    """
    cpu_count = os.cpu_count() if cpu_count is None else cpu_count
    partial_func = partial(calc_func, **func_kwargs)

    with mp.Pool(processes=cpu_count) as pool:
        results = []
        for result in tqdm(pool.imap(partial_func, indices, chunksize=chunk_size), total=len(indices)):
            results.append(result)
    return results

