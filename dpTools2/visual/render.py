from .utils import drawParticle, initialize_plot, drawVector, drawBoxBorders, config_anim_plot, create_pool_colors
from matplotlib import animation
import numpy as np
from tqdm import tqdm
from ..data import Data

import matplotlib.pyplot as plt
def draw_system(ax, pos, rad, box_size, draw_images, forces, draw_forces, color_array=None, **kwargs):
    for i in range(rad.size):
        # drawParticle(ax, pos[i], rad[i], **kwargs)
        if color_array is not None:
            color = color_array[i]
        else:
            color = 'black'
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

def update_animation(frame, ax_anim, frame_to_step, data: Data, which='particle', draw_forces=False, draw_center_particles=False):
    ax_anim.clear()
    if frame >= frame_to_step.size:
        frame = frame_to_step.size - 1
    step = frame_to_step[frame]
    index = np.where(data.trajectory.steps == step)[0][0]
    if not hasattr(data.trajectory[index], 'box_size'):
        box_size = data.system.box_size
    else:
        box_size = data.trajectory[index].box_size
    config_anim_plot(ax_anim, box_size, offset=0)
    drawBoxBorders(ax_anim, box_size, color='black', linestyle='--', alpha=0.5)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)  # Remove padding

    draw_frame(data, index, which=which, axes=[ax_anim], draw_forces=draw_forces, draw_center_particles=draw_center_particles)


def animate_data(data, num_frames, path, which="particle", draw_center_particles=False, draw_forces=False, num_end_frames=0, **kwargs):    
    if num_frames >= data.trajectory.steps.size:
        num_frames = data.trajectory.steps.size

    frame_to_step = data.trajectory.steps[::data.trajectory.steps.size // num_frames]

    # check if data[0] has box_size attribute:
    if not hasattr(data.trajectory[0], 'box_size'):
        fig, axes = initialize_plot(1, data.system.box_size, offset=0)
    else:
        fig, axes = initialize_plot(1, data.trajectory[0].box_size, offset=0)

    fig.tight_layout(pad=0)
    
    with tqdm(total=num_frames + num_end_frames, desc='Animating') as pbar:
        anim = animation.FuncAnimation(
            fig,
            update_animation,
            fargs=(axes[0], frame_to_step, data, which, draw_forces, draw_center_particles),
            frames=num_frames + num_end_frames,
            interval=100,
            blit=False
        )
        anim.save(path, progress_callback=lambda i, n: pbar.update(1), **kwargs)
        pbar.close()


def update_pool_animation(frame, ax_anim, frame_to_step, color_array, white_array, pos_indices, shuffled_indices, data: Data, which='particle', draw_forces=False):
    ax_anim.clear()
    step = frame_to_step[frame]
    index = np.where(data.trajectory.steps == step)[0][0]
    box_size = data.system.box_size
    config_anim_plot(ax_anim, box_size, offset=0, bg_color='#2B5329')
    drawBoxBorders(ax_anim, box_size, color='brown', linestyle='-', alpha=1, linewidth=5)
    plt.subplots_adjust(left=0.001, right=0.999, bottom=0.001, top=0.999)
    if which == 'vertex':
        num_vertices_in_particle = data.system.num_vertices_in_particle.copy().astype(int)
        vertex_positions = data.trajectory[index].vertex_positions.copy()
        vertex_radii = data.system.particle_config['vertex_radius'] * np.ones(vertex_positions.shape[0])
        vertex_color_array = color_array[np.repeat(np.arange(num_vertices_in_particle.size), num_vertices_in_particle)]
        draw_system(ax_anim, vertex_positions, vertex_radii, box_size, True, None, False, color_array=vertex_color_array)
    positions = data.trajectory[index].positions.copy()
    radii = data.system.radii.copy()
    if which == 'vertex':
        radii -= data.system.particle_config['vertex_radius']
    new_positions = positions.copy()
    # new_positions[pos_indices] = positions[shuffled_indices]
    draw_system(ax_anim, new_positions, radii, data.system.box_size, True, None, False, color_array=color_array)
    draw_system(ax_anim, new_positions[9:15], radii[9:15] / 2, data.system.box_size, True, None, False, color_array=white_array)


def animate_pool(data, num_frames, path, which="particle", draw_forces=False, **kwargs):    
    if num_frames >= data.trajectory.steps.size:
        num_frames = data.trajectory.steps.size

    frame_to_step = data.trajectory.steps[::data.trajectory.steps.size // num_frames]

    num_particles = data.trajectory[0].positions.shape[0]
    color_array = create_pool_colors(num_particles)
    solid_perm = np.random.permutation(6)
    color_array[1:7] = color_array[1:7][solid_perm]
    color_array[9:15] = color_array[9:15][solid_perm]
    pos_perm = np.random.permutation(13)
    pos_indices = np.concatenate([np.arange(1, 8), np.arange(9, 15)])
    shuffled_indices = pos_indices[pos_perm]
    white_array = np.ones((7, 3)) * 0.8

    fig, axes = initialize_plot(1, data.system.box_size, offset=0, bg_color='#2B5329')

    fig.tight_layout(pad=0)
    
    with tqdm(total=num_frames, desc='Animating') as pbar:
        anim = animation.FuncAnimation(
            fig,
            update_pool_animation,
            fargs=(axes[0], frame_to_step, color_array, white_array, pos_indices, shuffled_indices, data, which, draw_forces),
            frames=num_frames,
            interval=100,
            blit=False
        )
        anim.save(path, progress_callback=lambda i, n: pbar.update(1), **kwargs)
        pbar.close()

def draw_frame(data, frame, which='particle', axes=None, draw_images=True, draw_forces=False, draw_center_particles=False, **kwargs):
    if not hasattr(data.trajectory[frame], 'box_size'):
        box_size = data.system.box_size
    else:
        box_size = data.trajectory[frame].box_size

    if axes is None:
        fig, axes = initialize_plot(1, box_size, offset=0)

    if which == 'particle':
        positions = data.trajectory[frame].positions
        if not hasattr(data.trajectory[frame], 'radii'):
            radii = data.system.radii
    else:
        positions = data.trajectory[frame].vertex_positions
        radii = data.system.particle_config['vertex_radius'] * np.ones(positions.shape[0])

    # Calculate the wrapped positions
    wrapped_positions = np.mod(positions, box_size)

    if draw_forces:
        if which == 'particle':
            if hasattr(data.trajectory[frame], 'forces'):
                forces = data.trajectory[frame].forces
            else:
                forces = np.zeros_like(positions)
        else:
            if hasattr(data.trajectory[frame], 'vertex_forces'):
                forces = data.trajectory[frame].vertex_forces
            else:
                forces = np.zeros_like(positions)
    else:
        forces = None

    # if which == 'particle':
    #     color_array = data.trajectory[frame].static_particle_index
    # else:
    #     color_array = data.trajectory[frame].vertex_particle_index

    # Draw the main particles
    # draw_system(axes[0], wrapped_positions, radii, box_size, draw_images, **kwargs)
    # draw_system(axes[0], wrapped_positions, radii, box_size, draw_images, forces, draw_forces, **kwargs)
    draw_system(axes[0], wrapped_positions, radii, box_size, draw_images, forces, draw_forces, color_array=None, **kwargs)
    if draw_center_particles:
        particle_radii = data.system.radii.copy()
        central_radii = particle_radii - data.system.particle_config['vertex_radius']
        particle_positions = data.trajectory[frame].positions
        wrapped_particle_positions = np.mod(particle_positions, box_size)
        draw_system(axes[0], wrapped_particle_positions, central_radii, box_size, draw_images, None, False, color_array=None, **kwargs)
    return axes