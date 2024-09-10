import os
from .parameters import Parameters
from .metrics import Metrics
from .info import Info

class System:

    params_fname = 'params.dat'
    default_metrics = ['energy.dat']
    default_info = ['boxSize.dat', 'vertexMasses.dat', 'radii.dat', 'particleRadii.dat']
    additional_metrics = ['corrs.dat', 'pair_corrs.dat']
    additional_info = []

    def __init__(self, path, metrics=None, info=None):
        self.parameters = Parameters(os.path.join(path, self.params_fname))
        self.load_metrics(path, metrics)
        self.load_info(path, info)

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

    def load_info(self, path, info=None):
        info = self.default_info.extend(info) if info is not None else self.default_info
        for filename in info:
            info_path = os.path.join(path, filename)
            if os.path.exists(info_path):
                info_name = filename.split('.')[0]
                info = Info(info_path)
                setattr(self, info_name, info)
            else:
                print(f"Warning: Info file {info_path} not found.")
        for filename in self.additional_info:
            info_path = os.path.join(path, filename)
            if os.path.exists(info_path):
                info_name = filename.split('.')[0]
                info = Info(info_path)
                setattr(self, info_name, info)

    def __repr__(self):
        variables = {
            var: repr(getattr(self, var)) for var in dir(self)
            if not var.startswith('_') and not callable(getattr(self, var))
            and var not in ['default_info', 'default_metrics', 'params_fname', 'additional_info', 'additional_metrics']
        }
        variables_repr = "\n\t".join(variables.keys())
        return f"System(\n\t{variables_repr}\n)"
