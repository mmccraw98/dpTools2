import numpy as np
import os
from .configuration import Configuration

class Trajectory:

    index_prefix = 't'

    def __init__(self, path, load_all=True):
        index_names = os.listdir(path)
        self.index_paths = [os.path.join(path, name) for name in index_names]
        self.steps = np.array([int(name.split(self.index_prefix)[-1]) for name in index_names])
        order = np.argsort(self.steps)
        self.steps = self.steps[order]
        self.index_paths = np.array(self.index_paths)[order]
        self.index = np.arange(len(self.index_paths))
        self.configurations = self.load_full_trajectory() if load_all else None

    def load_full_trajectory(self):
        return np.array([Configuration(str(path)) for path in self.index_paths])

    def load_configuration(self, path):
        return Configuration(path)

    def __getitem__(self, index):
        if self.configurations is None:
            return self.load_configuration(str(self.index_paths[index]))
        else:
            return self.configurations[index]

    def __len__(self):
        return len(self.steps)

    def __repr__(self):
        config_repr = repr(self[0]).replace('\n', '\n\t\t')
        return f"Trajectory(\n\tnum_configs={len(self)},\n\t{config_repr}\n)"
