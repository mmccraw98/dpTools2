import json
import uuid

def get_default_disk_config(n_particles, packing_fraction, **kwargs):
    config = {
        'particle_type': 'Disk',
        'n_particles': n_particles,
        'packing_fraction': packing_fraction,
        'particle_dim_block': 256,
        'e_c': 1.0,
        'n_c': 2.0,
        'mass': 1.0,
        'size_ratio': 1.4,
        'count_ratio': 0.5,
        'seed': -1,
        'neighbor_list_config': {
            'neighbor_cutoff_multiplier': 1.5,
            'neighbor_displacement_multiplier': 0.2,
            'num_particles_per_cell': 8.0,
            'cell_displacement_multiplier': 0.5,
            'neighbor_list_update_method': 'cell'
        }
    }
    for key, value in kwargs.items():
        config[key] = value
    return config

def get_default_rigid_bumpy_config(n_particles, packing_fraction, rotation=True, n_vertices_per_small_particle=26, n_vertices_per_large_particle=36, segment_length_per_vertex_diameter=1.0, **kwargs):
    count_ratio = 0.5
    n_small_particles = int(n_particles * count_ratio)
    n_large_particles = n_particles - n_small_particles
    n_vertices = n_small_particles * n_vertices_per_small_particle + n_large_particles * n_vertices_per_large_particle
    config = {
        'particle_type': 'RigidBumpy',
        'n_particles': n_particles,
        'n_vertices': n_vertices,
        'packing_fraction': packing_fraction,
        'particle_dim_block': 256,
        'vertex_dim_block': 256,
        'segment_length_per_vertex_diameter': segment_length_per_vertex_diameter,
        'vertex_radius': 0,
        'n_vertices_per_small_particle': n_vertices_per_small_particle,
        'n_vertices_per_large_particle': n_vertices_per_large_particle,
        'size_ratio': 1.4,
        'count_ratio': count_ratio,
        'rotation': rotation,
        'e_c': 1.0,
        'n_c': 2.0,
        'mass': 1.0,
        'seed': -1,
        'neighbor_list_config': {
            'neighbor_cutoff_multiplier': 1.5,
            'neighbor_displacement_multiplier': 0.2,
            'num_particles_per_cell': 8.0,
            'cell_displacement_multiplier': 0.5,
            'vertex_neighbor_cutoff_multiplier': 1.5,
            'vertex_neighbor_displacement_multiplier': 0.2,
            'neighbor_list_update_method': 'cell'
        }
    }
    for key, value in kwargs.items():
        config[key] = value
    return config


def get_log_config_from_names_lin(log_names, num_steps, num_saves, group_name):
    config = {
        'log_names': log_names,
        'save_style': 'lin',
        'save_freq': int(num_steps / num_saves),
        'group_name': group_name,
        'reset_save_decade': 10,
        'min_save_decade': 10,
        'multiple': 0,
        'decade': 10
    }
    if config['save_freq'] == 0:
        config['save_freq'] = 1
    return config

def get_log_config_from_names_log(log_names, num_steps, num_saves, min_save_decade, group_name):
    config = {
        'log_names': log_names,
        'save_style': 'log',
        'reset_save_decade': int(num_steps / num_saves),
        'min_save_decade': min_save_decade,
        'group_name': group_name,
        'multiple': 0,
        'decade': 10,
        'save_freq': 1
    }
    if config['reset_save_decade'] == 0:
        config['reset_save_decade'] = 1
    return config

def get_log_config_from_names_lin_everyN(log_names, save_freq, group_name):
    config = {
        'log_names': log_names,
        'save_style': 'lin',
        'save_freq': save_freq,
        'group_name': group_name,
        'reset_save_decade': 10,
        'min_save_decade': 10,
        'multiple': 0,
        'decade': 10,
    }
    return config

def get_log_config_from_names(log_names, group_name):
    config = {
        'log_names': log_names,
        'group_name': group_name,
        'save_style': 'lin',
        'save_freq': 1,
        'reset_save_decade': 10,
        'min_save_decade': 10,
        'multiple': 0,
        'decade': 10
    }
    return config

def save_config(config, path):
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)
    
def get_n_char_uuid(n):
    """Return a string of length n from a UUID's hex representation."""
    return uuid.uuid4().hex[:n]