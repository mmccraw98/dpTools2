import os
import numpy as np
import pandas as pd
from typing import List
from ..data import Data
from ..analysis import aggregation, calculations

def calculate_pair_correlation_function(
    data: Data, 
    can_load_all: bool = False, 
    r_min: float = None, 
    r_max: float = None, 
    num_distance_bins: int = 500, 
    num_angle_bins: int = None,
    num_angle_axis_bins: int = None,
    radii_filter: np.ndarray = None,
    angle_bins: np.ndarray = None, 
    angle_axis_bins: np.ndarray = None, 
    angle_period: float = None, 
    angle_offsets: np.ndarray = None,
    backend: str = 'threadpool', 
    chunk_size: int = 10,
    frame: int = None
):
    data_was_fully_loaded = data.trajectory.frames is not None

    if not data_was_fully_loaded and can_load_all:
        data = Data(data.root, load_all=True)

    if backend not in ['threadpool', 'multiprocessing', 'serial']:
        raise ValueError(f'Backend {backend} not recognized')

    if radii_filter is None:
        radii_filter = [data.system.radii != np.nan]

    box_size = data.system.box_size
    r_min = max(data.system.radii.min() if r_min is None else r_min, 0)
    r_max = min(box_size.mean() / 2 if r_max is None else r_max, box_size.mean() / 2)

    num_particles = len(radii_filter[0])
    distance_bins = np.linspace(r_min, r_max, num_distance_bins)

    if angle_offsets is None:
        angle_offsets = np.zeros(num_particles)

    names = ['g', 'r']
    if angle_bins is not None:
        names.append('angle')
    if angle_axis_bins is not None:
        names.append('angle_axis')
    
    if frame is None:
        indices = data.trajectory.index.copy()
        res_with_edges = data.pair_corr_func(indices[0], filters=radii_filter, distance_bins=distance_bins, angle_bins=angle_bins, angle_axis_bins=angle_axis_bins, angle_period=angle_period, return_edges=True, angle_offsets=angle_offsets)
        edges = res_with_edges.data[1]
        if backend == 'multiprocessing':
            res = aggregation.calculate_parallel_multiprocessing(indices, data.pair_corr_func, chunk_size=chunk_size, filters=radii_filter, distance_bins=distance_bins, angle_bins=angle_bins, angle_axis_bins=angle_axis_bins, angle_period=angle_period, return_edges=False, angle_offsets=angle_offsets)
        elif backend == 'threadpool':
            res = aggregation.calculate_parallel_threadpool(indices, data.pair_corr_func, chunk_size=chunk_size, filters=radii_filter, distance_bins=distance_bins, angle_bins=angle_bins, angle_axis_bins=angle_axis_bins, angle_period=angle_period, return_edges=False, angle_offsets=angle_offsets)
        else:
            res = aggregation.calculate_serial(indices, data.pair_corr_func, filters=radii_filter, distance_bins=distance_bins, angle_bins=angle_bins, angle_axis_bins=angle_axis_bins, angle_period=angle_period, return_edges=False, angle_offsets=angle_offsets)
    else:
        res = data.pair_corr_func(frame, filters=radii_filter, distance_bins=distance_bins, angle_bins=angle_bins, angle_axis_bins=angle_axis_bins, angle_period=angle_period, return_edges=True, angle_offsets=angle_offsets)
        edges = res.data[1]

    histograms = [np.zeros(_.shape) for _ in res[0].data[0]]
    for hist in res:
        for i, h in enumerate(hist.data[0]):
            histograms[i] += h
    for h in histograms:
        h /= len(res)

    g, bins = calculations.normalize_histograms_r_theta_phi(histograms, edges, box_size, num_particles, N_angle_bins=num_angle_bins, N_angle_axis_bins=num_angle_axis_bins, angle_period=angle_period, filters=radii_filter)

    bin_names = [name for name in names[1:]]
    index = pd.MultiIndex.from_product(bins, names=bin_names)

    data_dict = {'g': g[0].flatten()}
    df = pd.DataFrame(data_dict, index=index).reset_index()
    return df

def get_first_peak_locations(pc):
    r_min = pc[pc.g > 0].r.min()
    r0 = pc.r[pc.g.argmax()]
    return r_min, r0

def calculate_all_pair_correlation_functions(
        data: Data,
        can_load_all: bool = True,
        num_bins: int = 1000,
        backend: str = 'threadpool',
        chunk_size: int = 10,
        overwrite: bool = False
):
    if not overwrite and os.path.exists(f'{data.root}/{data.system_path}/pair_corrs.csv'):
        return
    filters = [data.system.radii == data.system.radii.min(), data.system.radii == data.system.radii.max()]
    results = {}
    for filter, name in zip(filters, ['min', 'max']):
        pc_full = calculate_pair_correlation_function(data, can_load_all=can_load_all, radii_filter=[filter], num_distance_bins=num_bins, backend=backend, chunk_size=chunk_size)
        r_min, r_max = get_first_peak_locations(pc_full)

        pc_first_peak = calculate_pair_correlation_function(data, can_load_all=can_load_all, radii_filter=[filter], r_min=0.97 * r_min, r_max=1.2 * r_max, num_distance_bins=num_bins, backend=backend, chunk_size=chunk_size)

        r_trough = pc_full.r[np.argmin(pc_full.g[pc_full.r > r_max]) + np.argwhere(pc_full.r > r_max)[0][0]]

        pc_second_third_peak = calculate_pair_correlation_function(data, can_load_all=can_load_all, radii_filter=[filter], r_min=r_trough, r_max=3 * r_trough, num_distance_bins=num_bins, backend=backend, chunk_size=chunk_size)

        results.update({
            f'g_full_{name}': pc_full.g,
            f'r_full_{name}': pc_full.r,
            f'g_first_peak_{name}': pc_first_peak.g,
            f'r_first_peak_{name}': pc_first_peak.r,
            f'g_second_third_peak_{name}': pc_second_third_peak.g,
            f'r_second_third_peak_{name}': pc_second_third_peak.r,
        })

    pd.DataFrame(results).to_csv(f'{data.root}/{data.system_path}/pair_corrs.csv', index=False, sep=',')

def get_wave_vectors_filters_and_names(data: Data) -> tuple[np.ndarray, list[np.ndarray], list[str]]:
    pc = data.system.pair_corrs
    box_size = data.system.box_size
    radii = data.system.radii

    wave_vectors = 2 * np.pi / np.array([
        pc.r_first_peak_min[pc.g_first_peak_min.argmax()], 
        radii.min() * 2, 
        pc.r_first_peak_max[pc.g_first_peak_max.argmax()], 
        radii.max() * 2, 
        box_size.mean() / 2
    ])

    filters = [
        radii == radii.min(),
        radii == radii.min(),
        radii == radii.max(),
        radii == radii.max(),
        np.ones(radii.size, dtype=bool)
    ]

    names = [
        'isf_g0_min',
        'isf_sigma_min',
        'isf_g0_max',
        'isf_sigma_max',
        'isf_bulk'
    ]

    return wave_vectors, filters, names

def calculate_time_correlations(
        data: Data,
        can_load_all: bool = True,
        backend: str = 'threadpool',
        time_pair_style: str = 'log',
        chunk_size: int = 10,
        num_isf_angles: int = 10,
        vertex_counts: List[int] = [26, 36],
        overwrite: bool = False,
        just_msd: bool = False,
        angle_corrs: bool = True,
):
    if not overwrite and os.path.exists(f'{data.root}/{data.system_path}/corrs.csv'):
        return
    
    data_was_fully_loaded = data.trajectory.frames is not None

    if not data_was_fully_loaded and can_load_all:
        data = Data(data.root, load_all=True)

    if backend not in ['threadpool', 'multiprocessing', 'serial']:
        raise ValueError(f'Backend {backend} not recognized')
    
    if not hasattr(data.system, 'pair_corrs'):
        data = Data(data.root, load_all=True)
        data_was_fully_loaded = True
        if not hasattr(data.system, 'pair_corrs'):
            raise ValueError('Pair correlation functions must be calculated first')

    if time_pair_style == 'log':
        pairs = aggregation.get_log_pairs(data.trajectory.steps)
    else:
        pairs = aggregation.generate_all_time_pairs(data.trajectory.steps)

    wave_vectors, filters, names = get_wave_vectors_filters_and_names(data)
    if backend == 'multiprocessing':
        msd_results = aggregation.calculate_parallel_multiprocessing(pairs, data.msd_corr_func, chunk_size=chunk_size, drift_correction=True, particle_level=True)
    elif backend == 'threadpool':
        msd_results = aggregation.calculate_parallel_threadpool(pairs, data.msd_corr_func, chunk_size=chunk_size, drift_correction=True, particle_level=True)
    else:
        msd_results = aggregation.calculate_serial(pairs, data.msd_corr_func, drift_correction=True, particle_level=True)
    msd, time_lags = aggregation.average_by_time_lag(msd_results)

    corrs = {
        'msd': msd,
        'time_lags': time_lags,
    }

    if not just_msd:
        for name, filter, wave_vector in zip(names, filters, wave_vectors):
            if backend == 'multiprocessing':
                isf_results = aggregation.calculate_parallel_multiprocessing(pairs, data.self_isf_corr_func, chunk_size=chunk_size, wave_vector=wave_vector, filter=filter, num_angles=num_isf_angles, drift_correction=True, particle_level=True)
            elif backend == 'threadpool':
                isf_results = aggregation.calculate_parallel_threadpool(pairs, data.self_isf_corr_func, chunk_size=chunk_size, wave_vector=wave_vector, filter=filter, num_angles=num_isf_angles, drift_correction=True, particle_level=True)
            else:
                isf_results = aggregation.calculate_serial(pairs, data.self_isf_corr_func, wave_vector=wave_vector, filter=filter, num_angles=num_isf_angles, drift_correction=True, particle_level=True)
            isf, time_lags = aggregation.average_by_time_lag(isf_results)
            corrs[name] = isf
        
        if angle_corrs and hasattr(data.trajectory[0], 'angles'):
            if backend == 'multiprocessing':
                rot_msd_results = aggregation.calculate_parallel_multiprocessing(pairs, data.rot_msd_corr_func, chunk_size=chunk_size)
            elif backend == 'threadpool':
                rot_msd_results = aggregation.calculate_parallel_threadpool(pairs, data.rot_msd_corr_func, chunk_size=chunk_size)
            else:
                rot_msd_results = aggregation.calculate_serial(pairs, data.rot_msd_corr_func)
            
            rot_msd, time_lags = aggregation.average_by_time_lag(rot_msd_results)
            corrs['rot_msd'] = rot_msd

            for name, filter, num_vertex in zip(
                ['sigma_min', 'sigma_max'],
                [data.system.radii == data.system.radii.min(), data.system.radii == data.system.radii.max()],
                vertex_counts
            ):
                if backend == 'multiprocessing':
                    rot_isf_results = aggregation.calculate_parallel_multiprocessing(pairs, data.rot_self_isf_corr_func, chunk_size=chunk_size, filter=filter, n=num_vertex)
                elif backend == 'threadpool':
                    rot_isf_results = aggregation.calculate_parallel_threadpool(pairs, data.rot_self_isf_corr_func, chunk_size=chunk_size, filter=filter, n=num_vertex)
                else:
                    rot_isf_results = aggregation.calculate_serial(pairs, data.rot_self_isf_corr_func, filter=filter, n=num_vertex)
                rot_isf, time_lags = aggregation.average_by_time_lag(rot_isf_results)
                corrs[f'rot_self_isf_{name}'] = rot_isf

    pd.DataFrame(corrs).to_csv(f'{data.root}/{data.system_path}/corrs.csv', index=False, sep=',')
