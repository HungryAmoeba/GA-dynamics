# Geometric Algebra Simulations

## Setup Instructions

1. Clone the repository:
    ```sh
    git clone https://github.com/HungryAmoeba/GA-dynamics.git
    cd geom_algebra_sims
    ```

2. Create and activate a virtual environment:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Running the Main Program

To run the main program from the command line, use:
```sh
python main.py
```

This generates a simulation and stores a 3D rendering of the dynamics, representations, and the coordinates of the model. Several geometries are available for selection:

```yaml
geometry: 
    type: "helical"  # Options: "helical", "overhand_knot", "trefoil"

# General settings
fps: 30             # Frames per second
duration: 10        # Duration in seconds
output: "trajectories/trajectory.csv"  # Output file path

# Geometry-specific parameters
geometry_params:
    helical:
        num_nodes: 50
        orientation: "CCW"  # Options: "CCW" or "CW"
        radius: 1
        height: 1
        num_rotations: 2

    overhand_knot:
        num_nodes: 25
        num_nodes_linear_extension: 15
        scale: .5

    trefoil:
        num_nodes: 25
        scale: 2

    extension:
        num_nodes_head: 10
        num_nodes_tail: 10

# Camera settings and model positioning
positioning:
    positive_z: True

dynamics:
    head_push: 50
    tail_push: 50
    gradient_estimator: 3

# resampling parameters
```

### Overriding Config Parameters

You can override configuration parameters using Hydra. For example:
```sh
python main.py parameter_name=value
```
### Using a Different Configuration File

To use a different configuration file with Hydra, you can specify the path to the new config file using the `--config-path` and `--config-name` options. For example:

```sh
python main.py --config-path path/to/config --config-name new_config
```

This will override the default configuration with the settings from `new_config.yaml` located in the specified directory.

## Data Storage

By default, the data is saved in the `/data` directory. This information can be modified in `src/__init__.py`.
