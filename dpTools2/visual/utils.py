from matplotlib import patches, path, pyplot as plt, rcParams
from cycler import cycler
import matplotlib
import numpy as np

class RoundedPolygon(patches.PathPatch):
    # taken from: https://stackoverflow.com/questions/19270673/matplotlib-radius-in-polygon-edges-is-it-possible
    def __init__(self, xy, pad, **kwargs):
        """Draw a rounded polygon with matplotlib.
        
        Args:
            xy (list): list of vertices
            pad (float): padding
            **kwargs: Additional keyword arguments for the PathPatch class.
            
        Returns:
            None
        """
        p = path.Path(*self.__round(xy=xy, pad=pad))
        super().__init__(path=p, **kwargs)

    def __round(self, xy, pad):
        """Round the vertices of the polygon.
        
        Args:
            xy (list): list of vertices
            pad (float): padding
            
        Returns:
            list: list of rounded vertices
        """
        n = len(xy)

        for i in range(0, n):

            x0, x1, x2 = np.atleast_1d(xy[i - 1], xy[i], xy[(i + 1) % n])

            d01, d12 = x1 - x0, x2 - x1
            d01, d12 = d01 / np.linalg.norm(d01), d12 / np.linalg.norm(d12)

            x00 = x0 + pad * d01
            x01 = x1 - pad * d01
            x10 = x1 + pad * d12
            x11 = x2 - pad * d12

            if i == 0:
                verts = [x00, x01, x1, x10]
            else:
                verts += [x01, x1, x10]
        codes = [path.Path.MOVETO] + n*[path.Path.LINETO, path.Path.CURVE3, path.Path.CURVE3]

        return np.atleast_1d(verts, codes)

def drawParticle(ax_anim, pos, radius, **kwargs):
    """Draw a circle on the animation plot.
    
    Args:
        ax_anim (matplotlib.axes.Axes): Axes object to be configured.
        pos (list): Position of the particle [x, y].
        radius (float): Radius of the particle.
        color (str): Colour of the particle (default: 'black').
        **kwargs: Additional keyword arguments for the add_patch function.

    Returns:
        None
    """
    circle = plt.Circle(pos, radius, **kwargs)
    ax_anim.add_patch(circle)

def drawVector(ax_anim, pos, vector, tol=1e-6, **kwargs):
    if np.linalg.norm(vector) > tol:
        # ax_anim.quiver(pos, vector, angles='xy', scale_units='xy', scale=1, **kwargs)
        ax_anim.arrow(pos[0], pos[1], vector[0], vector[1], head_width=0.1, head_length=0.1, **kwargs)

def config_anim_plot(ax_anim, box_lengths, offset=3):
    """Configure the animation plot.
    
    Args:
        ax_anim (axes): axes object to be configured
        box_lengths (list): list of box lengths
        offset (int): offset of the box (default: 3)
        
        Returns:
            None
    """
    ax_anim.set_aspect('equal', adjustable='box')
    # set the limits to be the BOX_LENGTHS
    ax_anim.set_xlim(-offset, box_lengths[0] + offset)
    ax_anim.set_ylim(-offset, box_lengths[1] + offset)
    ax_anim.set_xticks([])
    ax_anim.set_yticks([])
    ax_anim.set_xticklabels([])
    ax_anim.set_yticklabels([])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)  # Remove padding

def initialize_plot(N, box_lengths, offset=3):
    """Initialize the plot.
    
    Args:
        N (int): number of subplots
        box_lengths (list): list of box lengths
        offset (int): offset of the box (default: 3)
        
        Returns:
            fig, axes (tuple): tuple of figure and axes objects
    """
    fig, axes = plt.subplots(1, N, figsize=(12, 6))
    if N == 1:
        axes = [axes]
    config_anim_plot(axes[0], box_lengths, offset=offset)
    return fig, axes

def drawBoxBorders(ax_anim, box_lengths, **kwargs):
    """Draw the box borders on the animation plot.
    
    Args:
        ax_anim (matplotlib.axes.Axes): Axes object to be configured.
        box_lengths (list): list of box lengths
        **kwargs: Additional keyword arguments for the add_patch function.
        
    Returns:
        None
    """
    # get the box lengths
    box_length_x, box_length_y = box_lengths

    # draw the box borders
    ax_anim.add_patch(
        patches.Rectangle(
            (0, 0),
            box_length_x,
            box_length_y,
            fill=False,
            **kwargs
        )
    )