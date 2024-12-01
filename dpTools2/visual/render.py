from .utils import drawParticle, initialize_plot, drawVector, drawBoxBorders, config_anim_plot
from matplotlib import animation
import numpy as np
from tqdm import tqdm
from ..data import Data

import matplotlib.pyplot as plt
def draw_system(ax, pos, rad, box_size, draw_images, forces, draw_forces, **kwargs):
    for i in range(rad.size):
        # drawParticle(ax, pos[i], rad[i], **kwargs)
        color = plt.cm.coolwarm(i / rad.size)
        # color = plt.cm.viridis(i / rad.size)
        drawParticle(ax, pos[i], rad[i], color=color)
        if draw_forces:
            drawVector(ax, pos[i], forces[i], tol=0.0)
        if draw_images:
            # check if pos[i] is less than rad[i] away from the border in any direction
            if np.any(pos[i] < rad[i]) or np.any(pos[i] > box_size - rad[i]):
                for x in [-1, 0, 1]:
                    for y in [-1, 0, 1]:
                        if x != 0 or y != 0:
                            # drawParticle(ax, pos[i] + box_size * np.array([x, y]), rad[i], **kwargs)
                            drawParticle(ax, pos[i] + box_size * np.array([x, y]), rad[i], color=color)

def draw_vector(ax, pos, vector, rad):
    for i in range(rad.size):
        drawVector(ax, pos[i], vector[i], tol=0.1)

def update_animation(frame, ax_anim, frame_to_step, data: Data, which='particle', draw_forces=False):
    ax_anim.clear()
    step = frame_to_step[frame]
    index = np.where(data.trajectory.steps == step)[0][0]
    if not hasattr(data.trajectory[index], 'boxSize'):
        box_size = data.system.boxSize
    else:
        box_size = data.trajectory[index].boxSize
    config_anim_plot(ax_anim, box_size, offset=0)
    drawBoxBorders(ax_anim, box_size, color='black', linestyle='--', alpha=0.5)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)  # Remove padding

    draw_frame(data, index, which=which, axes=[ax_anim], draw_forces=draw_forces)
    # if which == 'particle':
    #     pos = data[index].particle_positions
    #     rad = data.particle_radii
    # else:
    #     pos = data[index].positions
    #     rad = data.radii
    # # get a color map for the dpm ids
    # ids = np.arange(rad.size)
    # if len(ids) > 1:
    #     id_to_color = getValToColorMap(ids)
    #     # id_to_color = ['b', 'r', 'g', 'c']
    # else:
    #     id_to_color = getValToColorMap([0, 1])
    # pIds = np.arange(rad.size)
    # # for pid in pIds:
    #     # drawParticle(ax_anim, np.mod(pPos[step][pid], box_size), pRad[pid], color='r', alpha=0.5)
    # for pid in ids:
    #     color = id_to_color[pid]
    #     drawParticle(ax_anim, np.mod(pos[pid], box_size), rad[pid], color=color)
    # ax_anim.set_title(step)
    # ax_anim.set_title(data.phi[index])

def animate_data(data, num_frames, path, which="particle", draw_forces=False, **kwargs):    
    if num_frames >= data.trajectory.steps.size:
        num_frames = data.trajectory.steps.size

    frame_to_step = data.trajectory.steps[::data.trajectory.steps.size // num_frames]

    # check if data[0] has box_size attribute:
    if not hasattr(data.trajectory[0], 'boxSize'):
        fig, axes = initialize_plot(1, data.system.boxSize, offset=0)
    else:
        fig, axes = initialize_plot(1, data.trajectory[0].boxSize, offset=0)

    fig.tight_layout(pad=0)
    
    with tqdm(total=num_frames, desc='Animating') as pbar:
        anim = animation.FuncAnimation(
            fig,
            update_animation,
            fargs=(axes[0], frame_to_step, data, which, draw_forces),
            frames=num_frames,
            interval=100,
            blit=False
        )
        anim.save(path, progress_callback=lambda i, n: pbar.update(1), **kwargs)
        pbar.close()

def draw_frame(data, frame, which='particle', axes=None, draw_images=True, draw_forces=False, **kwargs):
    if not hasattr(data.trajectory[frame], 'boxSize'):
        box_size = data.system.boxSize
    else:
        box_size = data.trajectory[frame].boxSize

    if axes is None:
        fig, axes = initialize_plot(1, box_size, offset=0)

    if which == 'particle':
        positions = data.trajectory[frame].particlePos
        if not hasattr(data.trajectory[frame], 'particleRadii'):
            radii = data.system.particleRadii
        else:
            radii = data.trajectory[frame].particleRadii
    else:
        positions = data.trajectory[frame].positions
        if not hasattr(data.trajectory[frame], 'radii'):
            radii = data.system.radii
        else:
            radii = data.trajectory[frame].radii

    # Calculate the wrapped positions
    wrapped_positions = np.mod(positions, box_size)

    if hasattr(data.trajectory[frame], 'forces'):
        forces = data.trajectory[frame].forces
    else:
        forces = np.zeros_like(positions)

    # Draw the main particles
    # draw_system(axes[0], wrapped_positions, radii, box_size, draw_images, **kwargs)
    draw_system(axes[0], wrapped_positions, radii, box_size, draw_images, forces, draw_forces, **kwargs)
    return axes