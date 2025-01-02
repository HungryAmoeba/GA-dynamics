import os

import numpy as np
from einops import rearrange


class Representation:
    """
    Base class for representations of trajectories
    """

    def __init__(self, cfg, adjacency, trajectory, rep_dir, exp_save_name):
        self.cfg = cfg
        self.adjacency = adjacency
        self.trajectory = trajectory
        self.rep_dir = rep_dir
        self.exp_save_name = exp_save_name
        self.rep_type = cfg.representation.rep_type

    def generate(self):
        """Generate the representation. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement `generate` method.")

    def save(self, representation):
        """Save the representation to a file."""
        if self.cfg.save_representation:
            file_path = os.path.join(self.rep_dir, f"{self.rep_type}_{self.exp_save_name}.npy")
            np.save(file_path, representation)
            print(f"Representation saved at {file_path}")
