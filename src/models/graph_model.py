
def generate_graph_mujoco_xml(adjacency, position_dict=None):
    """
    Generate MuJoCo XML for a graph.
    
    Args:
        adjacency (dict or list): 
            - If dict: adjacency list where keys are nodes and values are lists of connected nodes.
            - If list: adjacency matrix (square list of lists or numpy array).
        position_dict (dict): Dictionary with node positions as {node: (x, y, z)}. Defaults to None.
            - If None, positions will be generated in a simple grid layout.
    
    Returns:
        str: MuJoCo XML string representing the graph.
    """
    # Handle adjacency list or matrix
    if isinstance(adjacency, list):
        from collections import defaultdict
        adjacency_list = defaultdict(list)
        for i, row in enumerate(adjacency):
            for j, value in enumerate(row):
                if value:  # Non-zero indicates an edge
                    adjacency_list[i].append(j)
    else:
        adjacency_list = adjacency

    # Handle positions
    if position_dict is None:
        position_dict = {
            node: (i, 0, 0) for i, node in enumerate(adjacency_list)
        }
    
    # Ensure all nodes in adjacency list are in position_dict
    for node in adjacency_list:
        if node not in position_dict:
            raise ValueError(f"Node {node} in adjacency list but not in position_dict")

    xml = '<mujoco>\n  <asset>\n'
    # add texture and material for the grid
    xml += '    <texture name="grid" type="2d" builtin="checker" rgb1="0.8 0.8 0.8" rgb2="0.4 0.4 0.4" width="512" height="512"/>\n'
    xml += '    <material name="grid" texture="grid" texrepeat="8 8" texuniform="true"/>\n'
    xml += '  </asset>\n'

    # Start worldbody
    xml += '  <worldbody>\n'
    
    # Add a light source
    xml += '    <light name="top" pos="0 0 10" dir="0 0 -1" diffuse="1 1 1"/>\n'

    # Add a visual grid on the floor
    xml += '    <geom type="plane" size="10 10 0.1" rgba="1 1 1 1" material="grid"/>\n'
    
    # Add nodes (bodies and geoms)
    for node, position in position_dict.items():
        xml += f'    <!-- Node {node} -->\n'
        xml += f'    <body name="{node}" pos="{position[0]} {position[1]} {position[2]}">\n'
        xml += f'      <geom name="geom_{node}" size="0.02" type="sphere" rgba="0.8 0.2 0.1 1"/>\n'
        xml += f'      <site name="site_{node}" pos="0 0 0" size="0.01"/>\n'
        xml += f'      <joint name="joint_{node}" type="free"/>\n'
        xml += '    </body>\n'

    # Close worldbody
    xml += '  </worldbody>\n'

    # Tendons section for edges
    xml += '    <!-- Tendon Definitions -->\n'
    xml += '  <tendon>\n'
    for node, neighbors in adjacency_list.items():
        for neighbor in neighbors:
            if node < neighbor:  # Avoid duplicating edges
                xml += f'    <!-- Edge {node}-{neighbor} -->\n'
                xml += f'    <spatial name="tendon_{node}_{neighbor}" width="0.01">\n'
                xml += f'      <site site="site_{node}"/>\n'
                xml += f'      <site site="site_{neighbor}"/>\n'
                xml += '    </spatial>\n'
    xml += '  </tendon>\n'

    xml += '  </mujoco>'

    return xml
