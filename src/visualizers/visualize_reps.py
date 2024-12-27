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


def make_spectrogram_movie(spectral_rep, eigenvalues, fps, use_abs=False, **kwargs):
    """
    Creates and saves an animated spectrogram with a global color bar for intensity.
    """
    from matplotlib import gridspec

    xlabel = kwargs.get("xlabel", "Time")
    ylabel = kwargs.get("ylabel", "Frequency (Eigenvalues)")
    title = kwargs.get("title", "Spectrogram")
    output_file = kwargs.get("output_file", "spectrogram_movie.mp4")

    T, N, D = spectral_rep.shape
    spectral_rep = np.abs(spectral_rep) if use_abs else spectral_rep ** 2

    fig = plt.figure(figsize=(10, 4 * D))
    gs = gridspec.GridSpec(D, 2, width_ratios=[50, 1], wspace=0.4)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(D)]
    cax = fig.add_subplot(gs[:, 1])

    ims = []
    ax_labels = ["x", "y", "z"]
    for i, ax in enumerate(axes):
        ax.set_title(f"{title} ({ax_labels[i]} coordinate)")
        ax.set_xlabel(xlabel if i == D - 1 else "")
        ax.set_ylabel(ylabel)
        ax.set_ylim(eigenvalues.min(), eigenvalues.max())
        im = ax.imshow(
            spectral_rep[0:1, :, i].T,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=[0, 1, eigenvalues.min(), eigenvalues.max()],
            animated=True,
            cmap="viridis",
        )
        ims.append(im)

    # Add a single color bar
    cbar = fig.colorbar(ims[0], cax=cax)
    cbar.set_label("Intensity")

    # Add dynamic time text
    time_text = fig.text(0.5, 0.02, "", ha="center", fontsize=12)

    def update(frame):
        for i, im in enumerate(ims):
            im.set_array(spectral_rep[: frame + 1, :, i].T)
            im.set_extent([0, frame + 1, eigenvalues.min(), eigenvalues.max()])
        time_text.set_text(f"Time: {frame}/{T}")
        return ims + [time_text]

    ani = animation.FuncAnimation(fig, update, frames=T, blit=True)
    ani.save(output_file, fps=fps, extra_args=["-vcodec", "libx264"])
    plt.close(fig)
    print(f"Spectrogram movie saved as {output_file}")
