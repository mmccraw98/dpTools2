import numpy as np

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

def computeDistances(pos, boxSize):
    numParticles = pos.shape[0]
    distances = (np.repeat(pos[:, np.newaxis, :], numParticles, axis=1) - np.repeat(pos[np.newaxis, :, :], numParticles, axis=0))
    distances += boxSize / 2
    distances %= boxSize
    distances -= boxSize / 2
    distances = np.sqrt(np.sum(distances**2, axis=2))
    return distances

def calc_pair_corr(pos, radii, box_size, n_bins=50):
    bins = np.linspace(radii.mean() / 10, box_size.mean() / 2, n_bins + 1)
    pbc_pos = getPBCPositions(pos, box_size)
    dists = computeDistances(pbc_pos, box_size)
    h, r = np.histogram(dists, bins=bins, density=False)
    r = 0.5 * (r[:-1] + r[1:])
    return r, h