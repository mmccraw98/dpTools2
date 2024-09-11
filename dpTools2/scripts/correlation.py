import os
import numpy as np
import pandas as pd
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
    backend: str = 'threadpool', 
    chunk_size: int = 10
):
    
    if angle_bins is not None or angle_axis_bins is not None:
        raise NotImplementedError('Angle correlation functions are not implemented yet')

    data_was_fully_loaded = data.trajectory.configurations is not None

    if not data_was_fully_loaded and can_load_all:
        data = Data(data.root, load_all=True)

    if backend not in ['threadpool', 'multiprocessing', 'serial']:
        raise ValueError(f'Backend {backend} not recognized')

    if radii_filter is None:
        radii_filter = [data.system.particleRadii != np.nan]

    box_size = data.system.boxSize
    r_min = max(data.system.particleRadii.min() if r_min is None else r_min, 0)
    r_max = min(box_size.mean() / 2 if r_max is None else r_max, box_size.mean() / 2)
    num_particles = len(radii_filter[0])
    distance_bins = np.linspace(r_min, r_max, num_distance_bins)
    
    names = ['g', 'r']
    if angle_bins is not None:
        names.append('angle')
    if angle_axis_bins is not None:
        names.append('angle_axis')

    indices = data.trajectory.index.copy()
    res_with_edges = data.pair_corr_func(indices[0], filters=radii_filter, distance_bins=distance_bins, angle_bins=angle_bins, angle_axis_bins=angle_axis_bins, angle_period=angle_period, return_edges=True)
    edges = res_with_edges.data[1]
    if backend == 'multiprocessing':
        res = aggregation.calculate_parallel_multiprocessing(indices, data.pair_corr_func, chunk_size=chunk_size, filters=radii_filter, distance_bins=distance_bins, angle_bins=angle_bins, angle_axis_bins=angle_axis_bins, angle_period=angle_period, return_edges=False)
    elif backend == 'threadpool':
        res = aggregation.calculate_parallel_threadpool(indices, data.pair_corr_func, chunk_size=chunk_size, filters=radii_filter, distance_bins=distance_bins, angle_bins=angle_bins, angle_axis_bins=angle_axis_bins, angle_period=angle_period, return_edges=False)
    else:
        res = aggregation.calculate_serial(indices, data.pair_corr_func, filters=radii_filter, distance_bins=distance_bins, angle_bins=angle_bins, angle_axis_bins=angle_axis_bins, angle_period=angle_period, return_edges=False)

    histograms = [np.zeros(len(_)) for _ in res[0].data[0]]
    for hist in res:
        for i, h in enumerate(hist.data[0]):
            histograms[i] += h
    for h in histograms:
        h /= len(res)
    g, bins = calculations.normalize_histograms_r_theta_phi(histograms, edges, box_size, num_particles, N_angle_bins=num_angle_bins, N_angle_axis_bins=num_angle_axis_bins, angle_period=angle_period, filters=radii_filter)

    if not data_was_fully_loaded and can_load_all:
        data = Data(data.root, load_all=False)

    return pd.DataFrame({'r': bins[0], 'g': g[0]})

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
    if not overwrite and os.path.exists(f'{data.root}/{data.system_path}/pair_corrs.dat'):
        return
    filters = [data.system.particleRadii == data.system.particleRadii.min(), data.system.particleRadii == data.system.particleRadii.max()]
    results = {}
    for filter, name in zip(filters, ['min', 'max']):
        pc_full = calculate_pair_correlation_function(data, can_load_all=can_load_all, radii_filter=[filter], num_distance_bins=num_bins, backend=backend, chunk_size=chunk_size)
        r_min, r_max = get_first_peak_locations(pc_full)

        pc_first_peak = calculate_pair_correlation_function(data, can_load_all=can_load_all, radii_filter=[filter], r_min=0.97 * r_min, r_max=1.2 * r_max, num_distance_bins=num_bins, backend=backend, chunk_size=chunk_size)

        pc_second_third_peak = calculate_pair_correlation_function(data, can_load_all=can_load_all, radii_filter=[filter], r_min=2 * 1.5 * r_min, r_max=2 * 4 * r_max, num_distance_bins=num_bins, backend=backend, chunk_size=chunk_size)

        results.update({
            f'g_full_{name}': pc_full.g,
            f'r_full_{name}': pc_full.r,
            f'g_first_peak_{name}': pc_first_peak.g,
            f'r_first_peak_{name}': pc_first_peak.r,
            f'g_second_third_peak_{name}': pc_second_third_peak.g,
            f'r_second_third_peak_{name}': pc_second_third_peak.r,
        })

    pd.DataFrame(results).to_csv(f'{data.root}/{data.system_path}/pair_corrs.dat', index=False, sep='\t')

def get_wave_vectors_filters_and_names(data: Data) -> tuple[np.ndarray, list[np.ndarray], list[str]]:
    pc = data.system.pair_corrs
    box_size = data.system.boxSize
    particle_radii = data.system.particleRadii

    wave_vectors = 2 * np.pi / np.array([
        pc.r_first_peak_min[pc.g_first_peak_min.argmax()], 
        particle_radii.min() * 2, 
        pc.r_first_peak_max[pc.g_first_peak_max.argmax()], 
        particle_radii.max() * 2, 
        box_size.mean() / 2
    ])

    filters = [
        particle_radii == particle_radii.min(),
        particle_radii == particle_radii.min(),
        particle_radii == particle_radii.max(),
        particle_radii == particle_radii.min(),
        np.ones(particle_radii.size, dtype=bool)
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
        overwrite: bool = False
):
    if not overwrite and os.path.exists(f'{data.root}/{data.system_path}/corrs.dat'):
        return
    
    data_was_fully_loaded = data.trajectory.configurations is not None

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
        pairs = aggregation.generate_logscheme_time_pairs(data.trajectory.index, 0, 1)
    else:
        pairs = aggregation.generate_all_time_pairs(data.trajectory.index)

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

    for name, filter, wave_vector in zip(names, filters, wave_vectors):
        if backend == 'multiprocessing':
            isf_results = aggregation.calculate_parallel_multiprocessing(pairs, data.self_isf_corr_func, chunk_size=chunk_size, wave_vector=wave_vector, filter=filter, num_angles=num_isf_angles, drift_correction=True, particle_level=True)
        elif backend == 'threadpool':
            isf_results = aggregation.calculate_parallel_threadpool(pairs, data.self_isf_corr_func, chunk_size=chunk_size, wave_vector=wave_vector, filter=filter, num_angles=num_isf_angles, drift_correction=True, particle_level=True)
        else:
            isf_results = aggregation.calculate_serial(pairs, data.self_isf_corr_func, wave_vector=wave_vector, filter=filter, num_angles=num_isf_angles, drift_correction=True, particle_level=True)
        isf, time_lags = aggregation.average_by_time_lag(isf_results)
        corrs[name] = isf

    pd.DataFrame(corrs).to_csv(f'{data.root}/{data.system_path}/corrs.dat', index=False, sep='\t')

    if not data_was_fully_loaded and can_load_all:
        data = Data(data.root, load_all=False)
