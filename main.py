import datetime
import json
import os

import hydra
import mujoco
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src import (
    CONFIGS_DIR,
    DATA_DIR,
    JOINED_VIDEOS_DIR,
    MODEL_FILES_DIR,
    REPRESENTATIONS_DIR,
    TRAJECTORIES_DIR,
    VIDEOS_DIR,
)
from src.dynamics.basic_movement import (
    animate_positions,
    constant_center_of_mass,
    positive_z,
    positive_z_trajectory,
    rotate_to_flat,
    rotate_to_flat_based_on_ends,
)
from src.dynamics.trajectory import TrajectoryResampler
from src.models.model import generate_model
from src.representations.laplacian_reps import (
    compress_spectral_rep,
    get_spectral_representation,
)
from src.representations.representation_factory import get_representation_instance
from src.visualizers.join_videos import join_videos_side_by_side
from src.visualizers.postprocessor import RepresentationPostProcessor
from src.visualizers.visualize_base import RepresentationVisualizer
from src.visualizers.visualize_reps import make_movie_of_reps, make_spectrogram_movie
from src.visualizers.visualize_trajectory import visualize_trajectory


def combined_name(cfg, log_dir=os.path.join(CONFIGS_DIR, "logs")):
    """
    Generates a simple name for the experiment based on the configuration
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    key_params = f"{cfg.geometry.type}_nodes{cfg.geometry.num_nodes}_{cfg.dynamics.type}-dynamics"
    experiment_name = f"{key_params}_{timestamp}"

    # Save the config to a file
    os.makedirs(log_dir, exist_ok=True)
    config_path = os.path.join(log_dir, f"{experiment_name}_config.json")
    # with open(config_path, "w") as f:
    #     json.dump(cfg, f, indent=4)

    return experiment_name


@hydra.main(version_base=None, config_path=CONFIGS_DIR, config_name="config")
def main(cfg: DictConfig):
    """
    Run the experiment based on the configuration
    """

    # Print loaded configuration for debugging
    print(OmegaConf.to_yaml(cfg))

    exp_save_name = combined_name(cfg)

    model, pos, adjacency = generate_model(
        cfg.geometry, os.path.join(MODEL_FILES_DIR, f"{exp_save_name}.xml"), **cfg
    )

    positions_list = animate_positions(pos, **cfg.dynamics)

    # Resample trajectory
    resampler = TrajectoryResampler(positions_list, framerate=cfg.fps, total_time=cfg.duration)
    trajectory = resampler.resample_trajectory()
    if cfg.save_trajectory:
        np.save(os.path.join(TRAJECTORIES_DIR, f"{exp_save_name}.npy"), trajectory)

    # see if there are any specified changes to location, i.e. constant center of mass
    if cfg.positioning.positive_z:
        trajectory = positive_z_trajectory(trajectory)
    if cfg.positioning.rotate_to_flat:
        trajectory = rotate_to_flat_based_on_ends(trajectory)
    if cfg.positioning.constant_center_of_mass:
        trajectory = constant_center_of_mass(trajectory, cfg.positioning.center_of_mass)
    if cfg.positioning.gaussian_noise:
        trajectory = trajectory + np.random.normal(
            0, cfg.positioning.gaussian_noise, trajectory.shape
        )

    # animate_positions(trajectory, fps=cfg.fps, duration=cfg.duration)
    visualize_trajectory(
        model,
        trajectory,
        save=True,
        filename=os.path.join(VIDEOS_DIR, f"{exp_save_name}.mp4"),
    )

    # make a directory for the representations using the experiment name
    rep_dir = os.path.join(REPRESENTATIONS_DIR, exp_save_name)
    os.makedirs(rep_dir, exist_ok=True)

    # Step 1: Generate Representation
    representation = get_representation_instance(
        cfg, adjacency, trajectory, rep_dir, exp_save_name
    )
    spectral_rep, eigenvalues = representation.generate()

    # Step 2: Visualize Representation
    visualizer = RepresentationVisualizer(cfg, rep_dir, exp_save_name)
    for dim_red_method in cfg.representation.dim_red_methods:
        print(f"Visualizing {dim_red_method} representation...")
        if dim_red_method == "spectrogram":
            if eigenvalues is not None:
                visualizer.visualize_spectrogram(spectral_rep, eigenvalues)
            else:
                print("Skipping spectrogram visualization: eigenvalues are not available.")
        else:
            visualizer.visualize_dim_reduction(spectral_rep, dim_red_method)

    # Step 3: Join Videos and Cleanup
    post_processor = RepresentationPostProcessor(cfg, JOINED_VIDEOS_DIR, exp_save_name)
    for dim_red_method in cfg.representation.dim_red_methods:
        representation_video = os.path.join(rep_dir, f"{dim_red_method}_{exp_save_name}.mp4")
        post_processor.join_videos(
            os.path.join(VIDEOS_DIR, f"{exp_save_name}.mp4"),
            representation_video,
            representation.rep_type,
            dim_red_method,
        )
        post_processor.cleanup([representation_video])

    post_processor.cleanup([os.path.join(VIDEOS_DIR, f"{exp_save_name}.mp4")])

    print("Done!")


if __name__ == "__main__":
    main()
