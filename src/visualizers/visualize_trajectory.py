import numpy as np
import mujoco
import mediapy as media

def visualize_trajectory(model, position_tensor, save = False, filename = None): 
    '''
    Visualizes a trajectory of positions in MuJoCo

    Inputs:
        model (mujoco.MjModel): MuJoCo model
        position_tensor (np.array): array of shape T x N x D where T is the number of frames, N is the number of nodes, and D is the dimension of the positions
    '''
    data = mujoco.MjData(model)
    # Initial conditions
    num_nodes = len(position_tensor[0])
    positions = np.zeros((num_nodes, 3))  # Store positions (x, y, z)
    for i, pos in enumerate(position_tensor[0]):
        positions[i] = pos

    for i, pos in enumerate(positions):
        data.qpos[7 * i:7 * i + 3] = pos
    # # Simulation parameters
    # duration = 5  # seconds
    framerate = 60  # Hz
    #dt = 1.0 / framerate  # Time step size

    pixels = []

    for i in range(len(position_tensor)):
        positions = position_tensor[i]
        # Directly update positions in MuJoCo
        for i, pos in enumerate(positions):
            data.qpos[7 * i:7 * i + 3] = pos
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model) as renderer:
            renderer.update_scene(data)
            pixels.append(renderer.render())
    media.show_video(pixels, fps = framerate)

    if save:
        media.write_video(filename, pixels, fps = framerate)

    return None

