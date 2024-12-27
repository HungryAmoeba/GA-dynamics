import mediapy as media
import mujoco
import numpy as np


def visualize_trajectory(model, position_tensor, save=False, filename=None, **kwargs):
    """
    Visualizes a trajectory of positions in MuJoCo

    Inputs:
        model (mujoco.MjModel): MuJoCo model
        position_tensor (np.array): array of shape T x N x D where T is the number of frames, N is the number of nodes, and D is the dimension of the positions
    """
    data = mujoco.MjData(model)
    # Initial conditions
    num_nodes = len(position_tensor[0])
    positions = np.zeros((num_nodes, 3))  # Store positions (x, y, z)
    for i, pos in enumerate(position_tensor[0]):
        positions[i] = pos

    for i, pos in enumerate(positions):
        data.qpos[7 * i : 7 * i + 3] = pos

    # Simulation parameters
    framerate = 60  # Hz
    pixels = []

    # Create a high-resolution renderer
    resolution = kwargs.get("resolution", (1420, 1080))  # Increase resolution (width, height)
    with mujoco.Renderer(model, width=resolution[0], height=resolution[1]) as renderer:
        for frame in position_tensor:
            # Update positions in MuJoCo
            for i, pos in enumerate(frame):
                data.qpos[7 * i : 7 * i + 3] = pos
            mujoco.mj_forward(model, data)

            # Update and render the scene
            renderer.update_scene(data)
            image = renderer.render()

            # Optional: Apply antialiasing or filters here if needed
            pixels.append(image)

    # Display the video
    media.show_video(pixels, fps=framerate)

    # Save the video if required
    if save and filename:
        media.write_video(filename, pixels, fps=framerate)

    return None
