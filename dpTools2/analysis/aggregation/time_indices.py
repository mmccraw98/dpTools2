import numpy as np
import numba as nb

def get_log_pairs(steps):
    indices = np.tril_indices(steps.size, k=-1)
    differences = (steps[:, None] - steps[None, :])[indices]
    oom_j = np.min(differences)
    pairs = []
    while True:
        for i in range(1, 10):
            mask = differences == oom_j * i
            pairs.extend(list(zip(indices[0][mask], indices[1][mask])))
        oom_j *= 10
        if np.sum(mask) == 0:
            break
    return pairs

@nb.njit(parallel=True, fastmath=True)
def generate_logscheme_time_pairs(timesteps: np.ndarray,
                                  starting_step: int = 0,
                                  freq_power: int = 1,
                                  step_decade_: int = 10) -> np.ndarray:
    """
    Finds all the pairs of time points (i, j) satisfying the relationship t(i) - t(j) = tau
    for a prescribed set of tau values.
    """
    max_power = int(np.ceil(np.log10(timesteps.max())))
    freq_decade = int(10 ** freq_power)
    num_blocks = int(10 ** (max_power - freq_power))
    decade_spacing = 10
    spacing_decade = 1
    step_decade = step_decade_

    # Start with a small array and grow it if needed
    all_pairs = np.zeros((int(1e4), 2), dtype=np.int64)
    count = 0

    # Use sets and dicts for faster lookup
    timestep_set = set(timesteps)
    time_indices = {time: idx for idx, time in enumerate(timesteps)}

    for power in range(max_power):
        for spacing in range(1, decade_spacing):
            step_increment = spacing * spacing_decade
            step_range = np.arange(0, step_decade, step_increment, dtype=np.int64)
            
            for multiple in range(starting_step, num_blocks):
                block_start = multiple * freq_decade
                for start, end in zip(step_range[:-1], step_range[1:]):
                    start_time = block_start + start
                    end_time = block_start + end
                    if start_time in timestep_set and end_time in timestep_set:
                        i = time_indices[start_time]
                        j = time_indices[end_time]
                        if count >= all_pairs.shape[0]:  # If array is full, grow it
                            all_pairs = np.resize(all_pairs, (all_pairs.shape[0] * 2, 2))
                        all_pairs[count, 0] = i
                        all_pairs[count, 1] = j
                        count += 1

        step_decade *= 10
        spacing_decade *= 10

    return all_pairs[:count]


@nb.njit(parallel=True, fastmath=True)
def generate_all_time_pairs(timesteps: np.ndarray) -> np.ndarray:
    """
    Generates all pairs (i, j) such that j < i from the given timesteps array.
    """
    num_timesteps = len(timesteps)
    num_pairs = (num_timesteps * (num_timesteps - 1)) // 2  # Total number of pairs
    all_pairs = np.zeros((num_pairs, 2), dtype=np.int64)  # Pre-allocate array

    count = 0
    for i in nb.prange(1, num_timesteps):
        for j in range(i):
            all_pairs[count, 0] = i
            all_pairs[count, 1] = j
            count += 1

    return all_pairs[:count]
