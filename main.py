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
from src.visualizers.join_videos import join_videos_side_by_side
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
        filename=os.path.join(VIDEOS_DIR, f"{exp_save_name}.mp4", **cfg),
    )

    # make a directory for the representations using the experiment name
    rep_dir = os.path.join(REPRESENTATIONS_DIR, exp_save_name)
    os.makedirs(rep_dir, exist_ok=True)

    # for now, only support one type of representation per job
    if cfg.representation.rep_type == "laplacian":
        representation_type = cfg.representation.rep_type
        spectral_rep_dict = get_spectral_representation(
            adjacency, trajectory, normalized=cfg.representation.use_normalized
        )  # TODO: make this take in cfg later
        spectral_rep = spectral_rep_dict["spectral_representation"]
        np.save(os.path.join(rep_dir, f"{representation_type}_{exp_save_name}.npy"), spectral_rep)
        # make and save a movie of the spectral representation
        # now compress the spectral representation
        for dim_red_method in cfg.representation.dim_red_methods:
            if dim_red_method == "spectogram":
                compressed_spectral_rep = spectral_rep
                eigenvalues = spectral_rep_dict["eigenvalues"]
                # see if filter zero eigenvalue
                if cfg.representation.filter_zero_eigenvalues:
                    print("Filtering zero eigenvalues")
                    compressed_spectral_rep = compressed_spectral_rep[:, 1:, :]
                    eigenvalues = eigenvalues[1:]
                make_spectrogram_movie(
                    spectral_rep,
                    eigenvalues,
                    fps=cfg.fps,
                    output_file=os.path.join(
                        rep_dir, f"{representation_type}_{dim_red_method}_{exp_save_name}.mp4"
                    ),
                    title="Spectrogram Representation",
                    use_abs=False,
                )
            else:
                compressed_spectral_rep = compress_spectral_rep(
                    spectral_rep,
                    dim_red_method=dim_red_method,
                    **{k: v for k, v in cfg.representation.items() if k != "dim_red_method"},
                )
                np.save(
                    os.path.join(
                        rep_dir, f"{representation_type}_{dim_red_method}_{exp_save_name}.npy"
                    ),
                    compressed_spectral_rep,
                )
                make_movie_of_reps(
                    compressed_spectral_rep,
                    fps=cfg.fps,
                    output_file=os.path.join(
                        rep_dir, f"{representation_type}_{dim_red_method}_{exp_save_name}.mp4"
                    ),
                    title=cfg.representation.get("title", "Representation Trajectory")
                    + f" ({dim_red_method})",
                )
            # join the videos side by side
            join_videos_side_by_side(
                os.path.join(VIDEOS_DIR, f"{exp_save_name}.mp4"),
                os.path.join(
                    rep_dir, f"{representation_type}_{dim_red_method}_{exp_save_name}.mp4"
                ),
                os.path.join(
                    JOINED_VIDEOS_DIR,
                    f"{exp_save_name}_{representation_type}_{dim_red_method}.mp4",
                ),
                fps=cfg.fps,
            )

    else:
        raise ValueError(f"Unknown representation type: {representation_type}")

    print("Done!")


if __name__ == "__main__":
    main()
