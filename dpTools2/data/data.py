import numpy as np
import os
from .system import System
from .trajectory import Trajectory
from ..analysis.aggregation import Result
class Data:

    system_path = 'system'
    trajectory_path = 'trajectories'

    def __init__(self, path, load_all=True):
        self.root = path
        if not os.path.exists(os.path.join(path, self.system_path)):
            try:
                self.system = System(path)
            except:
                raise ValueError(f"System path not found. "
                                 f"Contents of directory: {os.listdir(self.root)}")
        else:
            self.system = System(os.path.join(path, self.system_path))
        self.load_trajectory(os.path.join(path, self.trajectory_path), load_all)

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
        pass

    def __repr__(self):
        trajectory_repr = ''
        if hasattr(self, 'trajectory'):
            trajectory_repr = repr(self.trajectory).replace('\n', '\n\t\t')
        system_repr = repr(self.system).replace('\n', '\n\t\t')
        return f"Data(\n\tsystem={system_repr}\n\t{trajectory_repr}\n)"
    
    def msd_corr_func(self, pair, drift_correction=False, particle_level=False):
        i, j = pair
        if particle_level:
            delta = self.trajectory[i].particlePos - self.trajectory[j].particlePos
        else:
            delta = self.trajectory[i].positions - self.trajectory[j].positions
        if drift_correction:
            drift = np.mean(delta, axis=0)
            delta[:,0] -= drift[0]
            delta[:,1] -= drift[1]
        delta = np.linalg.norm(delta, axis=1)
        return Result(
            data=(np.mean(delta ** 2),),
            time=abs(self.trajectory.steps[i] - self.trajectory.steps[j])
        )
    