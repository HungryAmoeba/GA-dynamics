import os

import mujoco

from src.models.graph_mujoco_xml import generate_graph_mujoco_xml
from src.models.shapes import (
    get_ER_graph,
    get_helical_graph,
    get_overhand_knot,
    get_path_graph,
    get_trefoil_pos,
)


def generate_model(geometry_cfg, xml_path, **kwargs):
    """
    Generate a MuJoCo model based on the input parameters.

    Args:
        geometry_type (str): Type of geometry to generate.
        geometry_params (dict): Parameters for generating the geometry.
        extension_params (dict): Parameters for extending the geometry.
        positioning_params (dict): Parameters for positioning the geometry.
        dynamics_params (dict): Parameters for generating dynamics.

    Returns:
        mujoco.MjModel: MuJoCo model object.
    """

    # Geometry selection
    geom_type = geometry_cfg.type

    geometry_functions = {
        "helical": get_helical_graph,
        "overhand_knot": get_overhand_knot,
        "path": get_path_graph,
        "ER": get_ER_graph,
    }

    if geom_type in geometry_functions:
        func = geometry_functions[geom_type]
        pos, adjacency = func(**geometry_cfg)
    else:
        raise ValueError(f"Unknown geometry type: {geom_type}")

    # # Extend the graph and resample trajectory
    # pos, adjacency = extend_graph(graph, **extension_params)

    # if positioning_params.get('positive_z'):
    #     pos = positive_z(pos)

    # Generate MuJoCo XML
    resolution = kwargs.get("resolution", (1420, 1080))
    xml = generate_graph_mujoco_xml(adjacency, pos, resolution=resolution)
    # save the xml
    # xml_path = os.path.join(MODEL_FILES_DIR, f"{exp_save_name}.xml")
    with open(xml_path, "w") as f:
        f.write(xml)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return model, pos, adjacency
