from src.representations.laplacian_reps import LaplacianRepresentation
from src.representations.wavelets import SpectralWaveletRepresentation

REPRESENTATION_CLASSES = {
    "laplacian": LaplacianRepresentation,
    "spectral_wavelet": SpectralWaveletRepresentation,
    # "haar_wavelet": HaarWaveletRepresentation,
}


def get_representation_instance(cfg, adjacency, trajectory, rep_dir, exp_save_name):
    """
    Creates an instance of a representation class based on the configuration provided.
    Args:
        cfg (Config): Configuration object containing the representation type and other settings.
        adjacency (Any): Adjacency matrix or structure used in the representation.
        trajectory (Any): Trajectory data used in the representation.
        rep_dir (str): Directory where representation data is stored.
        exp_save_name (str): Name under which the experiment results will be saved.
    Returns:
        Representation: An instance of the representation class specified in the configuration.
    Raises:
        ValueError: If the representation type specified in the configuration is unknown.
    """

    rep_type = cfg.representation.rep_type
    if rep_type not in REPRESENTATION_CLASSES:
        raise ValueError(f"Unknown representation type: {rep_type}")
    return REPRESENTATION_CLASSES[rep_type](cfg, adjacency, trajectory, rep_dir, exp_save_name)
