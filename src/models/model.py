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
    Generates a MuJoCo model based on the specified geometry configuration and saves the model to an XML file.

    Args:
        geometry_cfg (object): Configuration object containing the type and parameters of the geometry.
        xml_path (str): Path where the generated XML file will be saved.
        **kwargs: Additional keyword arguments.
            - resolution (tuple): Resolution for the MuJoCo XML generation. Default is (1420, 1080).

    Returns:
        model: the MuJoCo model,
        pos (dict): positions of the nodes,
        adjacency (dict): An adjacency dictionary representing the graph structure.

    Raises:
        ValueError: If the geometry type specified in geometry_cfg is unknown.
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
