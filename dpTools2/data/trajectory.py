import numpy as np
from tqdm import tqdm
import os
from .frame import Frame

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
        self.frames = self.load_full_trajectory() if load_all else None

    def load_full_trajectory(self):
        return np.array([Frame(str(path)) for path in tqdm(self.index_paths)])

    def load_frame(self, path):
        return Frame(path)
    
    def resize(self, size):
        if size < len(self):
            self.frames = self.frames[:size]
            self.index = self.index[:size]
            self.steps = self.steps[:size]
            self.index_paths = self.index_paths[:size]
        else:
            self.frames = np.concatenate([self.frames, np.array([Frame(None)] * (size - len(self)))])
            self.index = np.arange(size)
            self.steps = np.concatenate([self.steps, np.array([None] * (size - len(self)))])
            self.index_paths = np.concatenate([self.index_paths, np.array([None] * (size - len(self)))])

    def __getitem__(self, index):
        if self.frames is None:
            return self.load_frame(str(self.index_paths[index]))
        else:
            return self.frames[index]

    def __len__(self):
        return len(self.steps)

    def __repr__(self):
        frame_repr = repr(self[0]).replace('\n', '\n\t\t')
        return f"Trajectory(\n\tnum_frames={len(self)},\n\t{frame_repr}\n)"
