# Geometric Algebra Simulations

> [!WARNING]  
> README is still under construction! 

## Setup Instructions

1. Clone the repository:

   ```sh
   git clone https://github.com/HungryAmoeba/GA-dynamics.git
   cd geom_algebra_sims
   ```

2. Create and activate a virtual environment:

   ```sh
   conda create -n geom
   conda activate geom
   ```

3. Install the required dependencies:

   ```sh
   pip install -r requirements.txt
   ```

## Running the Main Program

This codebase makes use of [hydra](https://hydra.cc) to manage configurations and configuration files. To run the main program from the command line with the default parameters and geometries, simply run:

```sh
python main.py
```

This runs the simulation and (optionally) saves the coordinates over time, a 3D rendering of the dynamics, representations of the dynamics with respect to a chosen basis, and the model. These are stored in data/. Logs and overrides are stored in outputs/. The standard configs is stored in [config.yaml](./configs/config.yaml).

### Geometries

Models can be specified from any arbitrary graph topology. This work can take as input any [networkx](https://networkx.org) graph. For convenience, several model geometries configs are given in [the geometry subdirectory](./configs/geometry/):

#### ER graph

This generates an Erdos-Renyi random graph. It has two parameters which need to be specified in the config:

- num_nodes - number of nodes
- p - probability of connecting two nodes

#### Helix graph

A helix graph is a path graph, and the starting node positioning is based on a parameterized curve.

For $\\theta_i \\in \[0, 2\\pi)$,

$$
\\text{pos}[\\theta_i] = \\left( r \\cos(\\theta_i), \\pm r \\sin(\\theta_i), \\theta_i \\right)
$$

where the interval $\[0, 2\\pi)$ is uniformly spaced samples based on the number of nodes and the sign of the second term is determined by the user specified orientation of the helix.

Parameters:

- num_nodes
- num_linear_extension - the number of nodes to add on to the end of the helix to extend the graph.

#### Overhand knot

The original positioning of the overhand knot is based on a parameterized curve describing a trefoil knot.

The positions coordinates are given by:

$$pos[t] = \\left(\\sin(t) + 2 \\sin(2t), \\cos(t) - 2 \\cos(2t), -\\sin(3t) \\right)$$

where $t$ is sampled from an interval contained in $\[0,2 \\pi)$ to ensure that the ends are accessible. In practice, $[0, 5.5\\pi/4]$ is used as the interval to sample from. Then the loose ends are extended in a straight line.

Parameters:

- num_nodes - the number of nodes to generate positions for.
- upper_limit - the upper limit for the parameter (t), which controls how close to being closed the trefoil knot is.
- scale - a scaling factor to adjust the size of the trefoil knot.
- num_nodes_head - the number of nodes to extend in the direction of the head (node 0)
- num_nodes_tail - the number of nodes to extend in the direction of the tail (node 0)

#### Path

Initial positioning is from uniformly spaced samples on the x-axis.

Parameters:

- num_nodes

### Dynamics

#### Follower

Follower dynamics apply to path graphs. The geometries defined above specify a static structure. This structure is animated using two types of motions. The first motion moves all nodes towards the front of the path graph. Specifically, the gradient $\\Delta_1$ at the front of the graph is attained by averaging differences in position between the first $k$ pairs of adjacent nodes.

For nodes $v_2, v_3, \\ldots v_n$ the position $x(v_k) := x(v\_{k - 1})$. Then the first position is updated $x(v_1) := x(v_1) + \\Delta_1$.

This procedure is iterated to produce an animation. An analogous procedure may be applied at the other end of the graph to simulate motion in that direction.

Linear interpolation is performed between these snapshots to generate smooth dynamics.

### Overriding Config Parameters

You can override configuration parameters using Hydra. For example:

```sh
python main.py geometry.type=overhand_knot geometry_params.extension.num_nodes_head=0 geometry_params.extension.num_nodes_tail=0
```

### Using a Different Configuration File

To use a different configuration file with Hydra, you can specify the path to the new config file using the `--config-path` and `--config-name` options. For example:

```sh
python main.py --config-path path/to/config --config-name new_config
```

This will override the default configuration with the settings from `new_config.yaml` located in the specified directory.

## Data Storage

By default, the data is saved in the `/data` directory. This information can be modified in `src/__init__.py`.
