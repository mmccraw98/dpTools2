from tqdm import tqdm
import warnings
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import os
from data import Data
from ..utils.io import recursive_walk_with_stopping_content

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

class BaseDataProcessor:  # base class for all data processors, modifies data by reference
    def __new__(cls):
        instance = super().__new__(cls)
        return instance
    
    def __call__(self, data, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

class BaseDFProcessor:  # base class for all data processors that modifies a global dataframe by reference
    def __new__(cls):
        instance = super().__new__(cls)
        return instance
    
    def __call__(self, df_to_process, global_df, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
    
class DFTimeAverager(BaseDFProcessor):  # time average a dataframe and append it to the global dataframe
    name = 'DFTimeAverager'
    def __call__(self, df_to_process, global_df, **kwargs):
        time_avg = df_to_process.mean()
        if len(global_df.columns) == 0:  # If DataFrame is empty, initialize with columns
            global_df[time_avg.index] = None  # Create columns first
        global_df.loc[len(global_df)] = time_avg  # Add the row to global_df in place

class DataSet:  # simple container for a list of Data objects, supporting filtering operations like numpy/pandas
    def __init__(
            self,
            root,  # location where all data is stored
            load_kwargs: Dict[str, Any] = {},  # kwargs for loading the individual data objects
            data_procs: List[BaseDataProcessor] = [],  # additional processing steps to apply to the data (extra loading / analysis)
            data_procs_kwargs: List[Dict[str, Any]] = [],  # kwargs for the additional processing steps
            df_processors: List[BaseDFProcessor] = [],  # additional processing steps to apply to the dataframe
            df_processors_kwargs: List[Dict[str, Any]] = [],  # kwargs for the additional processing steps
            ):
        self.root = root
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            paths = recursive_walk_with_stopping_content(root, ['system', 'system.zip', 'trajectories', 'trajectories.zip'])
            self.data = []
            for path in tqdm(paths, desc='Loading data', total=len(paths)):
                try:
                    self.data.append(Data(path, extract=True, **load_kwargs))
                except Exception as e:
                    print(f"Error loading {path}: {e}")
            self.process_data(data_procs, data_procs_kwargs)
        self.names = [os.path.basename(data.root) for data in self.data]
        self.config_df = self.get_config_keys()
        self.scalar_df = self.get_scalar_keys([DFTimeAverager()] + df_processors, [{}] + df_processors_kwargs)

    def process_data(
            self,
            data_procs: List[BaseDataProcessor] = [],
            data_procs_kwargs: List[Dict[str, Any]] = []
        ):
        for data_proc, data_proc_kwargs in zip(data_procs, data_procs_kwargs):
            for data in tqdm(self.data, desc=f'Processing {data_proc.name}', total=self.size()):
                data_proc(data, **data_proc_kwargs)

    def get_config_keys(
            self,
            extra_config_names: List[str] = [],
        ):
        config_names = ['run_config', 'integrator_config', 'particle_config'] + extra_config_names
        config_keys = pd.DataFrame()
        for data in self.data:
            row_data = {}
            for config_name in config_names:
                if hasattr(data.system, config_name):
                    config_dict = getattr(data.system, config_name)
                    row_data.update(flatten_dict(config_dict))
            new_row = pd.DataFrame([row_data])
            config_keys = pd.concat([config_keys, new_row], ignore_index=True)
        return config_keys
    
    def get_scalar_keys(
            self,
            df_processors: List[BaseDFProcessor] = [],
            df_processors_kwargs: List[Dict[str, Any]] = [],
        ):
        scalar_keys = pd.DataFrame()
        for df_processor, df_processor_kwargs in zip(df_processors, df_processors_kwargs):
            for data in tqdm(self.data, desc=f'Processing {df_processor.name}', total=self.size()):
                df_processor(data.system.energy, global_df=scalar_keys, **df_processor_kwargs)
        return scalar_keys
    
    def scalars(self):
        return self.scalar_df.copy()
    
    def configs(self):
        return self.config_df.copy()
    
    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)
    
    def size(self):
        return len(self.data)
    
    def __getitem__(self, key):
        # Handle simple integer/slice indexing of self.data
        if isinstance(key, (int, slice)):
            return self.data[key]
        
        # Handle boolean indexing or column-based conditions
        if isinstance(key, (pd.Series, np.ndarray)) and key.dtype == bool:
            # If we're given a boolean mask directly
            mask = key
        else:
            # Try to evaluate the condition on both DataFrames
            config_mask = pd.Series(True, index=range(len(self.data)))
            scalar_mask = pd.Series(True, index=range(len(self.data)))
            
            try:
                # Check if any column names in the condition exist in config_df
                config_cols = [col for col in self.config_df.columns if str(col) in str(key)]
                if config_cols:
                    config_mask = eval(f"self.config_df{str(key)}")
            except:
                pass
                
            try:
                # Check if any column names in the condition exist in scalar_df
                scalar_cols = [col for col in self.scalar_df.columns if str(col) in str(key)]
                if scalar_cols:
                    scalar_mask = eval(f"self.scalar_df{str(key)}")
            except:
                pass
            
            # Combine masks
            mask = config_mask & scalar_mask
        
        # Create new DataSet with filtered data
        new_ds = DataSet.__new__(DataSet)  # Create new instance without calling __init__
        new_ds.root = self.root
        new_ds.data = [d for i, d in enumerate(self.data) if mask[i]]
        new_ds.names = [self.names[i] for i, m in enumerate(mask) if m]
        new_ds.config_df = self.config_df[mask].reset_index(drop=True)
        new_ds.scalar_df = self.scalar_df[mask].reset_index(drop=True)
        return new_ds
    
    def __getattr__(self, name):
        if name in self.config_df.columns:
            return self.config_df[name]
        elif name in self.scalar_df.columns:
            return self.scalar_df[name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
    def __repr__(self):
        base_repr = f"DataSet(root={self.root}, size={len(self)}"
        if self.config_df.shape[0] > 0 and 'particle_type' in self.config_df.columns:
            pvcs = self.config_df['particle_type'].value_counts()
            base_repr += f", Particle Types={pvcs.to_dict()}"
        base_repr += ")"
        return base_repr