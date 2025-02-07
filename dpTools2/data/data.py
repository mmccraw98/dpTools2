import numpy as np
import os
from typing import List, Tuple
from ..utils.io.general import compress, decompress
from .system import System
from .trajectory import Trajectory
from ..analysis.aggregation import Result
from ..analysis.calculations import getPBCPositions, computeDistances

class Data:

    system_path = 'system'
    trajectory_path = 'trajectories'

    def __init__(self, path, load_all=True, load_trajectory=False, extract=True):
        self.root = path
        if self.system_is_compressed():
            if extract:
                self.decompress('system')
                self.load_system(path)
            else:
                print('System is compressed!')
                self.load_system(None)
        else:
            self.load_system(path)
        if load_trajectory:
            if self.trajectory_is_compressed():
                self.decompress('trajectories')
            self.load_trajectory(os.path.join(path, self.trajectory_path), load_all)

    def system_is_compressed(self):
        return os.path.exists(os.path.join(self.root, self.system_path + '.zip'))

    def trajectory_is_compressed(self):
        return os.path.exists(os.path.join(self.root, self.trajectory_path + '.zip'))

    def compress(self, *args):
        compress(self.root, *args)

    def decompress(self, *args):
        decompress(self.root, *args)

    def load_system(self, path):
        if path is None:
            self.system = System(None)
        elif not os.path.exists(os.path.join(path, self.system_path)):
            try:
                self.system = System(path)
            except:
                raise ValueError(f"System path not found. "
                                 f"Contents of directory: {os.listdir(self.root)}")
        else:
            self.system = System(os.path.join(path, self.system_path))

    def load_trajectory(self, path, load_all):
        if os.path.exists(path):
            self.trajectory = Trajectory(path, load_all)
        else:
            try:
                self.trajectory = Trajectory(self.root, load_all)
            except:
                raise ValueError(f"Trajectory path not found. "
                                 f"Contents of directory: {os.listdir(self.root)}")

    def validate_trajectory(self):
        pass  # TODO: Implement

    def __repr__(self):
        trajectory_repr = ''
        if hasattr(self, 'trajectory'):
            trajectory_repr = repr(self.trajectory).replace('\n', '\n\t\t')
        system_repr = repr(self.system).replace('\n', '\n\t\t')
        return f"Data(\n\tsystem={system_repr}\n\t{trajectory_repr}\n)"
    
    def msd_corr_func(self, pair: Tuple[int, int], drift_correction: bool = True, particle_level: bool = True) -> Result:
        i, j = pair
        if particle_level:
            delta = self.trajectory[i].positions - self.trajectory[j].positions
        else:
            delta = self.trajectory[i].vertex_positions - self.trajectory[j].vertex_positions
        if drift_correction:
            delta -= np.mean(delta, axis=0, keepdims=True)
        delta_squared = np.sum(delta ** 2, axis=1)
        return Result(
            data=(np.mean(delta_squared),),
            time=abs(self.trajectory.steps[i] - self.trajectory.steps[j])
        )

    def rot_msd_corr_func(self, pair: Tuple[int, int]) -> Result:
        i, j = pair
        angle_diff = self.trajectory[i].angles - self.trajectory[j].angles
        return Result(
            data=(np.mean(angle_diff ** 2),),
            time=abs(self.trajectory.steps[i] - self.trajectory.steps[j])
        )

    
    def self_isf_corr_func(self, pair: Tuple[int, int], wave_vector: np.ndarray, filter: np.ndarray = None, num_angles: int = 10, drift_correction: bool = True, particle_level: bool = True, backend: str = 'numpy') -> Result:
        i, j = pair
        if particle_level:
            delta = self.trajectory[i].positions - self.trajectory[j].positions
        else:
            delta = self.trajectory[i].vertex_positions - self.trajectory[j].vertex_positions
        if filter is not None:
            delta = delta[filter]

        if drift_correction:
            delta -= np.mean(delta, axis=0, keepdims=True)

        if backend == 'numpy':
            angle_list = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
            cos_angles = np.cos(angle_list)
            sin_angles = np.sin(angle_list)
            sq = np.mean(np.exp(1j * wave_vector * (delta[:, 0][:, None] * cos_angles + delta[:, 1][:, None] * sin_angles)), axis=0)
        else:
            sq = []
            angle_list = np.arange(0, 2 * np.pi, np.pi / num_angles)
            for angle in angle_list:
                q = np.array([np.cos(angle), np.sin(angle)])
                sq.append(np.mean(np.exp(1j * wave_vector * np.sum(np.multiply(q, delta), axis=1))))
            sq = np.array(sq)

        return Result(
            data=(np.real(np.mean(sq)),),
            time=abs(self.trajectory.steps[i] - self.trajectory.steps[j])
        )
    
    def rot_self_isf_corr_func(self, pair: Tuple[int, int], filter: np.ndarray = None, n: int = 1) -> Result:
        i, j = pair
        angle_diff = (self.trajectory[i].angles - self.trajectory[j].angles)
        if filter is not None:
            angle_diff = angle_diff[filter]
        return Result(
            data=(np.real(np.mean(np.exp(1j * n * angle_diff))),),
            time=abs(self.trajectory.steps[i] - self.trajectory.steps[j])
        )

    def pair_corr_func(self, i: int, filters: List[np.ndarray], distance_bins: np.ndarray, angle_bins: np.ndarray = None, angle_axis_bins: np.ndarray = None, angle_period: float = None, return_edges: bool = False, angle_offsets: np.ndarray = None) -> Result:
        pos = self.trajectory[i].positions
        bins = [distance_bins]
        if angle_axis_bins is not None and angle_period is not None:
            particle_angles = self.trajectory[i].angles
            if angle_offsets is not None:
                particle_angles += angle_offsets
            bins.append(angle_axis_bins)

        hist = []
        for f in filters:
            pbc_pos = getPBCPositions(pos[f], self.system.box_size)

            distances, diff = computeDistances(pbc_pos, self.system.box_size, return_diffs=True)

            sample = [distances]
            if angle_bins is not None and angle_period is not None:
                sample.append(np.arctan2(diff[:, :, 1], diff[:, :, 0]) % angle_period)
                bins.append(angle_bins)
            if angle_axis_bins is not None and angle_period is not None:
                sample.append(((particle_angles[f][:, np.newaxis] - particle_angles[f][np.newaxis, :]) % angle_period))
            mask = distances > 0

            hist_new, edges = np.histogramdd(
                [s[mask].flatten() for s in sample],
                bins=bins,
                density=False
            )
            hist.append(hist_new)
        if return_edges:
            return Result(
                data=(hist, edges),
            time=self.trajectory.steps[i]
        )
        else:
            return Result(
                data=(hist,),
                time=self.trajectory.steps[i]
            )
