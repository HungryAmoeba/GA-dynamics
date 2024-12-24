import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def make_movie_of_reps(reps, fps, **kwargs):
    """
    This function makes a movie of the given representations.

    Inputs:
        reps (np.array): array of shape T x 2 representing the trajectory in some representation
        fps (int): frames per second
        kwargs (dict): optional arguments for customization:
            - 'xlabel' (str): label for the x-axis (default: 'DIM1')
            - 'ylabel' (str): label for the y-axis (default: 'DIM2')
            - 'title' (str): title of the plot (default: 'Representation Trajectory')
            - 'output_file' (str): file name for saving the movie (default: 'trajectory_movie.mp4')
    """
    # Set default labels and output file name
    xlabel = kwargs.get("xlabel", "DIM1")
    ylabel = kwargs.get("ylabel", "DIM2")
    title = kwargs.get("title", "Representation Trajectory")
    output_file = kwargs.get("output_file", "trajectory_movie.mp4")

    # Set up the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(np.min(reps[:, 0]) - 1, np.max(reps[:, 0]) + 1)
    ax.set_ylim(np.min(reps[:, 1]) - 1, np.max(reps[:, 1]) + 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="both", which="both", length=0)  # Remove ticks
    # remove axis lines
    ax.grid(kwargs.get("grid", False))  # Add grid if specified

    # Initialize a scatter plot
    scatter = ax.scatter([], [], s=50)

    # Update function for animation
    def update(frame):
        scatter.set_offsets(reps[: frame + 1])  # Show up to the current frame
        return (scatter,)

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(reps), blit=True)

    # Save animation as .mp4
    ani.save(output_file, fps=fps, extra_args=["-vcodec", "libx264"])
    plt.close(fig)  # Close the figure after saving

    print(f"Movie saved as {output_file}")
