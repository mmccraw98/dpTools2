from .utils import drawParticle, initialize_plot, drawVector, drawBoxBorders, config_anim_plot
from matplotlib import animation
import numpy as np
from tqdm import tqdm
from ..data import Data

def draw_system(ax, pos, rad, box_size, draw_images, **kwargs):
    for i in range(rad.size):
        drawParticle(ax, pos[i], rad[i], **kwargs)
        if draw_images:
            # check if pos[i] is less than rad[i] away from the border in any direction
            if np.any(pos[i] < rad[i]) or np.any(pos[i] > box_size - rad[i]):
                for x in [-1, 0, 1]:
                    for y in [-1, 0, 1]:
                        if x != 0 or y != 0:
                            drawParticle(ax, pos[i] + box_size * np.array([x, y]), rad[i], **kwargs)

def draw_vector(ax, pos, vector, rad):
    for i in range(rad.size):
        drawVector(ax, pos[i], vector[i], tol=0.1)

def update_animation(frame, ax_anim, frame_to_step, data: Data, which='particle'):
    ax_anim.clear()
    step = frame_to_step[frame]
    index = np.where(data.trajectory.steps == step)[0][0]
    if not hasattr(data.trajectory[index], 'boxSize'):
        box_size = data.system.boxSize
    else:
        box_size = data.trajectory[index].boxSize
    config_anim_plot(ax_anim, box_size, offset=0)
    drawBoxBorders(ax_anim, box_size, color='black', linestyle='--', alpha=0.5)

    draw_frame(data, index, which=which, axes=[ax_anim])
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
    ax_anim.set_title(step)
    # ax_anim.set_title(data.phi[index])

def animate_data(data, num_frames, path, which="particle", **kwargs):    
    if num_frames >= data.trajectory.steps.size:
        num_frames = data.trajectory.steps.size

    frame_to_step = data.trajectory.steps[::data.trajectory.steps.size // num_frames]

    # check if data[0] has box_size attribute:
    if not hasattr(data.trajectory[0], 'boxSize'):
        fig, axes = initialize_plot(1, data.system.boxSize, offset=0)
    else:
        fig, axes = initialize_plot(1, data.trajectory[0].boxSize, offset=0)
    
    with tqdm(total=num_frames, desc='Animating') as pbar:
        anim = animation.FuncAnimation(
            fig,
            update_animation,
            fargs=(axes[0], frame_to_step, data, which),
            frames=num_frames,
            interval=100,
            blit=False
        )
        anim.save(path, progress_callback=lambda i, n: pbar.update(1), **kwargs)
        pbar.close()

def draw_frame(data, frame, which='particle', axes=None, draw_images=True, **kwargs):
    if not hasattr(data.trajectory[frame], 'boxSize'):
        box_size = data.system.boxSize
    else:
        box_size = data.trajectory[frame].boxSize

    if axes is None:
        fig, axes = initialize_plot(1, box_size, offset=0)

    if which == 'particle':
        positions = data.trajectory[frame].particlePos
        radii = data.system.particleRadii
    else:
        positions = data.trajectory[frame].positions
        radii = data.system.radii

    # Calculate the wrapped positions
    wrapped_positions = np.mod(positions, box_size)

    # Draw the main particles
    draw_system(axes[0], wrapped_positions, radii, box_size, draw_images, **kwargs)
    return axes