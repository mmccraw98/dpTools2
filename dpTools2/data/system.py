import os
from .parameters import Parameters
from .metrics import Metrics
from .info import Info
from ..utils.io import listdir

class System:

    default_metrics = ['energy.csv']
    additional_metrics = ['corrs.csv', 'pair_corrs.csv']

    def __init__(self, path, metrics=None):
        if path is not None:
            self.load_params(path)
            self.load_metrics(path, metrics)
            self.load_init(os.path.join(path, 'init'))

    def load_params(self, path):
        paths = listdir(path, full=True, files_only=True, file_types=['json'])
        for path in paths:
            name = os.path.basename(path).split('.')[0]
            parameters = Parameters(path, name=name)
            setattr(self, name, parameters)

    def load_metrics(self, path, metrics=None):
        if metrics is not None:
            metrics = self.default_metrics.extend(metrics)
        else:
            metrics = self.default_metrics
        for filename in metrics:
            metric_path = os.path.join(path, filename)
            if os.path.exists(metric_path):
                metric_name = filename.split('.')[0]
                metric = Metrics(metric_path)
                setattr(self, metric_name, metric)
            else:
                print(f"Warning: Metric file {metric_path} not found.")
        for filename in self.additional_metrics:
            metric_path = os.path.join(path, filename)
            if os.path.exists(metric_path):
                metric_name = filename.split('.')[0]
                metric = Metrics(metric_path)
                setattr(self, metric_name, metric)

    def load_init(self, path):
        for filename in listdir(path, full=True, files_only=True, file_types=['dat']):
            info_name = os.path.basename(filename).split('.')[0]
            info = Info(filename)
            setattr(self, info_name, info)

    def load_restart(self, path):
        self.restart = Restart()
        if path is None:
            return
        elif not os.path.exists(path):
            raise ValueError(f"Restart path {path} not found.")
        else:
            for filename in listdir(path, full=True, files_only=True, file_types=['dat']):
                info_name = os.path.basename(filename).split('.')[0]
                info = Info(filename)
                setattr(self.restart, info_name, info)

    def __repr__(self):
        variables = {
            var: repr(getattr(self, var)) for var in dir(self)
            if not var.startswith('_') and not callable(getattr(self, var))
            and var not in ['default_info', 'default_metrics', 'params_fname', 'additional_info', 'additional_metrics']
        }
        variables_repr = "\n\t".join(variables.keys())
        return f"System(\n\t{variables_repr}\n)"

class Restart:
    def __repr__(self):
        variables = {
            var: repr(getattr(self, var)) for var in dir(self)
            if not var.startswith('_') and not callable(getattr(self, var))
        }
        variables_repr = "\n\t".join(variables.keys())
        return f"Restart(\n\t{variables_repr}\n)"