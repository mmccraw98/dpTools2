import numpy as np
from typing import List, Tuple, Optional
def calculate_momentum_tensor(velocities: np.ndarray, masses: np.ndarray, dim: int = 2) -> np.ndarray:
    """Calculate the momentum tensor for a set of particles.
    
    Args:
        velocities (np.ndarray): The velocities of the particles.
        masses (np.ndarray): The masses of the particles.
        dim (int): The dimension of the system. Default is 2.
        
    Returns:
        np.ndarray: The momentum tensor.
    """
    momentum = velocities * masses[:, None]
    tensor = np.zeros(dim * dim)
    for i in range(dim):
        for j in range(dim):
            tensor[i * dim + j] = np.sum(momentum[:, i] * momentum[:, j])
    return tensor

def calc_particle_velocities(vel, numVertexInParticleList):
    cs = np.cumsum(numVertexInParticleList).astype(int)
    return np.array([np.mean(vel[i: j, :], axis=0) for i, j in zip(np.insert(cs[:-1], 0, 0), cs)])


def calc_distance_pbc(r1: np.ndarray, r2: np.ndarray, boxSize: np.ndarray) -> np.ndarray:
    delta = r1 - r2
    delta -= boxSize * np.round(delta / boxSize)
    return delta

def getPBCPositions(pos, boxSize):
    pos[:,0] -= np.floor(pos[:,0] / boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1] / boxSize[1]) * boxSize[1]
    return pos

def computeDistances(pos, boxSize, return_diffs=False):
    numParticles = pos.shape[0]
    diffs = (np.repeat(pos[:, np.newaxis, :], numParticles, axis=1) - np.repeat(pos[np.newaxis, :, :], numParticles, axis=0))
    diffs += boxSize / 2
    diffs %= boxSize
    diffs -= boxSize / 2
    distances = np.sqrt(np.sum(diffs ** 2, axis=2))
    if return_diffs:
        return distances, diffs
    return distances

def calc_pair_corr(pos, radii, box_size, n_bins=50):
    bins = np.linspace(radii.mean() / 10, box_size.mean() / 2, n_bins + 1)
    pbc_pos = getPBCPositions(pos, box_size)
    dists = computeDistances(pbc_pos, box_size)
    h, r = np.histogram(dists, bins=bins, density=False)
    r = 0.5 * (r[:-1] + r[1:])
    return r, h

def normalize_histograms_r_theta_phi(H: List[np.ndarray],
                                     edges: List[np.ndarray],
                                     box_size: np.ndarray,
                                     num_particles: int,
                                     N_angle_bins: Optional[int] = None,
                                     N_angle_axis_bins: Optional[int] = None,
                                     angle_period: Optional[float] = np.pi,
                                     filters: Optional[List[np.ndarray]] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Normalize histograms for any combination of distances, angles between particles, and relative orientation angles of particles.
    Can also filter particles based on particle-level criteria (i.e. minimum radius).
    
    Parameters:
        H: List[np.ndarray] - d-dimensional histograms
        edges: List[np.ndarray] - d-entry list of the edges of the histograms
        box_size: np.ndarray - box size
        num_particles: int - number of particles
        N_angle_bins: Optional[int] - number of bins for the angle between particles
        N_angle_axis_bins: Optional[int] - number of bins for the angle of particle j in the frame of particle i
        angle_period: Optional[float] - period of the angles
        filters: np.ndarray - particle-level filtering to make histograms for specific particles (i.e. minimum radius)
        
    Returns:
        List[np.ndarray]: List of d-dimensional normalized histograms (one for each filter)
        List[np.ndarray]: d-entry list of the centers of the bins
    """

    edge_centers = [(e[1:] + e[:-1]) / 2 for e in edges]
    differential_elements = [np.diff(e).mean() for e in edges]
    r = edge_centers[0]
    dr = differential_elements[0]

    G = []
    for h, f in zip(H, filters):
        num_particles = f.sum()
        number_density = num_particles / np.prod(box_size)

        # Calculate ideal distribution for distances (spherical shells)
        n_r_ideal = number_density * np.pi * ((r + dr) ** 2 - (r) ** 2)  # n_r_ideal should not depend on theta or phi

        if N_angle_bins is not None and N_angle_axis_bins is not None:
            constant = (num_particles * differential_elements[1] * differential_elements[2]) / (angle_period * angle_period)
            n_r_ideal = n_r_ideal[:, np.newaxis, np.newaxis]
        elif N_angle_bins is not None:
            constant = (num_particles * differential_elements[1]) / angle_period
            n_r_ideal = n_r_ideal[:, np.newaxis]
        elif N_angle_axis_bins is not None:
            constant = (num_particles * differential_elements[1]) / angle_period
            n_r_ideal = n_r_ideal[:, np.newaxis]
        else:
            constant = num_particles

        G.append(h / (n_r_ideal * constant))
    
    return G, edge_centers