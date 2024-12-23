import os
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import datetime
import json

from src import DATA_DIR, MODEL_FILES_DIR, REPRESENTATIONS_DIR, TRAJECTORIES_DIR, CONFIGS_DIR, VIDEOS_DIR
from src.models.graph_model import generate_graph_mujoco_xml
from src.models.shapes import get_helical_graph, extend_graph, get_overhand_knot, get_trefoil_pos
from src.visualizers.visualize_trajectory import visualize_trajectory
from src.dynamics.trajectory import TrajectoryResampler
from src.dynamics.basic_movement import animate_positions, constant_center_of_mass, positive_z
import mujoco

def combined_name(cfg, log_dir=os.path.join(CONFIGS_DIR, 'logs')):
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
    if cfg.geometry.type == 'helical':
        graph, _ = get_helical_graph(**cfg.geometry_params.helical)
    elif cfg.geometry.type == 'overhand_knot':
        graph, _ = get_overhand_knot(**cfg.geometry_params.overhand_knot)
    elif cfg.geometry.type == 'trefoil':
        graph, _ = get_trefoil_pos(**cfg.geometry_params.trefoil)
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
    with open(xml_path, 'w') as f:
        f.write(xml)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    
    # Animate and visualize
    positions_list = animate_positions(pos, **cfg.dynamics)

    # Resample trajectory
    resampler = TrajectoryResampler(positions_list, framerate=cfg.fps, total_time=cfg.duration)
    trajectory = resampler.resample_trajectory()
    np.save(os.path.join(TRAJECTORIES_DIR, f"{exp_save_name}.npy"), trajectory)

    #animate_positions(trajectory, fps=cfg.fps, duration=cfg.duration)
    visualize_trajectory(model, 
                         trajectory,
                         save=True,
                         filename=os.path.join(VIDEOS_DIR, f"{exp_save_name}.mp4"))

    # Save trajectory to output
    # output_path = os.path.join(REPRESENTATIONS_DIR, cfg.output)
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    #np.savetxt(output_path, trajectory, delimiter=',')
    print('Done!')
    #print(f"Trajectory saved to: {output_path}")


if __name__ == '__main__':
    main()
