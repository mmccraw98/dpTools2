import numpy as np
from typing import List, Tuple
from collections import defaultdict
from .result import Result

def sort_by_time(results: List[Result]) -> Tuple[np.ndarray, ...]:
    """
    Aggregates the 'data' tuples from a list of Result objects and sorts them based on 'time'.
    
    Parameters
    ----------
    results : List[Result]
        List of Result objects, where each object has 'data' (tuple) and 'time' (float or int).
        
    Returns
    -------
    tuple
        A tuple of numpy arrays: (val_1, val_2, ..., val_N, time_lags) sorted by time.
    """
    data_list = [result.data for result in results]
    time = np.array([result.time for result in results])

    data_array = np.array(data_list)

    values = [data_array[:, i] for i in range(data_array.shape[1])]
    
    sort_idx = np.argsort(time)
    sorted_time = time[sort_idx]
    sorted_values = [v[sort_idx] for v in values]

    return (*sorted_values, sorted_time)


def average_by_time_lag(results: List[Result]) -> Tuple[np.ndarray, ...]:
    """
    Aggregates and averages the 'data' tuples from a list of Result objects by unique 'time-lag' values,
    and sorts them based on 'time-lag'.
    
    Parameters
    ----------
    results : List[Result]
        List of Result objects, where each object has 'data' (tuple) and 'time' (float or int).
        
    Returns
    -------
    tuple
        A tuple of numpy arrays: (val_1, val_2, ..., val_N, time_lags) sorted by time_lag, where
        each val_i is the averaged value for the unique time_lag.
    """
    grouped_data = defaultdict(lambda: {"sum": None, "count": 0})

    for result in results:
        time_lag = result.time
        data = np.array(result.data)
        
        if grouped_data[time_lag]["sum"] is None:
            grouped_data[time_lag]["sum"] = data
        else:
            grouped_data[time_lag]["sum"] += data
        
        grouped_data[time_lag]["count"] += 1

    time_lags = []
    averaged_values = []

    for time_lag, data_dict in grouped_data.items():
        avg_data = data_dict["sum"] / data_dict["count"]
        time_lags.append(time_lag)
        averaged_values.append(avg_data)

    time_lags = np.array(time_lags)
    averaged_values = np.array(averaged_values)

    sort_idx = np.argsort(time_lags)
    sorted_time_lags = time_lags[sort_idx]
    sorted_averaged_values = averaged_values[sort_idx]

    sorted_values = [sorted_averaged_values[:, i] for i in range(sorted_averaged_values.shape[1])]

    return (*sorted_values, sorted_time_lags)
