import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


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


def make_oriented_area_movie(area_rep, fps, **kwargs):
    """
    Creates an animated movie showing the oriented area representation over time,
    with a color bar for the actual area values.

    Args:
        area_rep (np.array): Array of shape (T, N, 3) representing oriented area values.
        fps (int): Frames per second for the animation.
        **kwargs: Additional arguments, e.g., output_file.
    """
    # raise a ValueError if the area representation is not of shape T x N x 3
    if len(area_rep.shape) != 3 or area_rep.shape[2] != 3:
        raise ValueError("The area representation must be of shape (T, N, 3).")

    T, N, _ = area_rep.shape

    max_abs_area = np.max(np.abs(area_rep))
    if np.isclose(max_abs_area, 0):
        # make a warning
        print("The maximum absolute area value is close to zero. The plot may not be informative.")
        normalized_area = area_rep
    else:
        # Normalize the areas to be between -1 and 1 for sizes
        normalized_area = area_rep / max_abs_area

    # Find the min and max of the actual area values
    area_min, area_max = np.min(area_rep), np.max(area_rep)

    # Set up the figure and axes (3 stacked subplots)
    fig, axs = plt.subplots(3, 1, figsize=(15, 8))
    fig.suptitle("Oriented Area Representation")

    for i, ax in enumerate(axs):
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Sigma{i+1}")

    # Static x-positions for nodes, y=0 for all rows
    x = np.linspace(-0.8, 0.8, N)
    y = np.zeros(N)

    # Create scatter plots for each axis
    scatter_plots = [axs[i].scatter(x, y, s=50) for i in range(3)]

    # Create a color map and normalizer for the color bar
    norm = Normalize(vmin=area_min, vmax=area_max)
    cmap = plt.cm.coolwarm  # Blue to red colormap
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # Add a color bar to the figure
    cbar = fig.colorbar(sm, ax=axs, orientation="vertical", fraction=0.02, pad=0.05)
    cbar.set_label("Oriented Area Value")

    def init():
        """Initialize scatter plots."""
        for scatter in scatter_plots:
            scatter.set_sizes(np.zeros(N))
            scatter.set_color("gray")
        return scatter_plots

    def update(frame):
        """Update scatter plots for each frame."""
        for i, scatter in enumerate(scatter_plots):
            # Update sizes based on normalized values
            sizes = 200 * np.abs(normalized_area[frame, :, i])
            # Update colors based on actual area values
            colors = cmap(norm(area_rep[frame, :, i]))
            scatter.set_sizes(sizes)
            scatter.set_color(colors)
        return scatter_plots

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, init_func=init, frames=T, interval=1000 / fps, blit=True
    )

    output_file = kwargs.get("output_file", "oriented_area_movie.mp4")
    ani.save(output_file, fps=fps, extra_args=["-vcodec", "libx264"])
    plt.close(fig)
    print(f"Oriented area movie saved as {output_file}")
