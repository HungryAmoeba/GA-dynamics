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
    rotate_to_flat,
)
from src.dynamics.trajectory import TrajectoryResampler
from src.models.graph_model import generate_graph_mujoco_xml
from src.models.shapes import (
    extend_graph,
    get_helical_graph,
    get_overhand_knot,
    get_path_graph,
    get_trefoil_pos,
)
from src.representations.laplacian_reps import (
    compress_spectral_rep,
    get_spectral_representation,
)
from src.visualizers.join_videos import join_videos_side_by_side
from src.visualizers.visualize_reps import make_movie_of_reps
from src.visualizers.visualize_trajectory import visualize_trajectory


def combined_name(cfg, log_dir=os.path.join(CONFIGS_DIR, "logs")):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    key_params = f"{cfg.geometry.type}_nodes{cfg.geometry_params[cfg.geometry.type].num_nodes}"
    experiment_name = f"{key_params}_{timestamp}"

    # Save the config to a file
    os.makedirs(log_dir, exist_ok=True)
    config_path = os.path.join(log_dir, f"{experiment_name}_config.json")
    # with open(config_path, "w") as f:
    #     json.dump(cfg, f, indent=4)

    return experiment_name


@hydra.main(version_base=None, config_path=CONFIGS_DIR, config_name="config")
def main(cfg: DictConfig):
    # Print loaded configuration for debugging
    print(OmegaConf.to_yaml(cfg))

    # Geometry selection
    if cfg.geometry.type == "helical":
        graph, _ = get_helical_graph(**cfg.geometry_params.helical)
    elif cfg.geometry.type == "overhand_knot":
        graph, _ = get_overhand_knot(**cfg.geometry_params.overhand_knot)
    elif cfg.geometry.type == "trefoil":
        graph = get_trefoil_pos(**cfg.geometry_params.trefoil)
    elif cfg.geometry.type == "path":
        graph, _ = get_path_graph(**cfg.geometry_params.path)
    else:
        raise ValueError(f"Unknown geometry type: {cfg.geometry.type}")

    exp_save_name = combined_name(cfg)

    # Extend the graph and resample trajectory
    pos, adjacency = extend_graph(graph, **cfg.geometry_params.extension)

    if cfg.positioning.positive_z:
        pos = positive_z(pos)

    # Generate MuJoCo XML
    xml = generate_graph_mujoco_xml(adjacency, pos)
    # save the xml
    xml_path = os.path.join(MODEL_FILES_DIR, f"{exp_save_name}.xml")
    with open(xml_path, "w") as f:
        f.write(xml)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Animate and visualize
    positions_list = animate_positions(pos, **cfg.dynamics)

    # Resample trajectory
    resampler = TrajectoryResampler(positions_list, framerate=cfg.fps, total_time=cfg.duration)
    trajectory = resampler.resample_trajectory()
    np.save(os.path.join(TRAJECTORIES_DIR, f"{exp_save_name}.npy"), trajectory)

    # see if there are any specified changes to location, i.e. constant center of mass
    if cfg.positioning.constant_center_of_mass:
        trajectory = constant_center_of_mass(trajectory, cfg.positioning.center_of_mass)
    if cfg.positioning.rotate_to_flat:
        trajectory = rotate_to_flat(trajectory)
    if cfg.positioning.gaussian_noise:
        trajectory = trajectory + np.random.normal(
            0, cfg.positioning.gaussian_noise, trajectory.shape
        )

    # animate_positions(trajectory, fps=cfg.fps, duration=cfg.duration)
    visualize_trajectory(
        model, trajectory, save=True, filename=os.path.join(VIDEOS_DIR, f"{exp_save_name}.mp4")
    )

    # # get Laplacian spectral representation
    # spectral_rep = get_spectral_representation(adjacency, trajectory)
    # np.save(os.path.join(REPRESENTATIONS_DIR, f"{exp_save_name}_laplacian.npy"), spectral_rep)

    # make a directory for the representations using the experiment name
    rep_dir = os.path.join(REPRESENTATIONS_DIR, exp_save_name)
    os.makedirs(rep_dir, exist_ok=True)

    representation_types = cfg.representation.keys()
    for representation_type in representation_types:
        if cfg.representation[representation_type].rep_type == "laplacian":
            spectral_rep = get_spectral_representation(adjacency, trajectory)
            np.save(
                os.path.join(rep_dir, f"{representation_type}_{exp_save_name}.npy"), spectral_rep
            )
            # make and save a movie of the spectral representation
            # now compress the spectral representation
            compressed_spectral_rep = compress_spectral_rep(
                spectral_rep, **cfg.representation[representation_type]
            )
            np.save(
                os.path.join(rep_dir, f"{representation_type}_compressed_{exp_save_name}.npy"),
                compressed_spectral_rep,
            )
            make_movie_of_reps(
                compressed_spectral_rep,
                fps=cfg.fps,
                output_file=os.path.join(rep_dir, f"{representation_type}_{exp_save_name}.mp4"),
                title=cfg.representation[representation_type].get(
                    "title", "Representation Trajectory"
                ),
            )
        else:
            raise ValueError(f"Unknown representation type: {representation_type}")

        # join the videos side by side
        join_videos_side_by_side(
            os.path.join(VIDEOS_DIR, f"{exp_save_name}.mp4"),
            os.path.join(rep_dir, f"{representation_type}_{exp_save_name}.mp4"),
            os.path.join(JOINED_VIDEOS_DIR, f"{exp_save_name}_{representation_type}.mp4"),
        )

    print("Done!")


if __name__ == "__main__":
    main()
