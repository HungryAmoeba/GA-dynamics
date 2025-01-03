import numpy as np


class GeometricNumbers:
    def __init__(self):
        self.components = {
            "e1": np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
            "e2": np.array([[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, -1, 0, 0]]),
            "e3": np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]),
        }

        self.components["sigma1"] = self.components["e2"] @ self.components["e3"]
        self.components["sigma2"] = self.components["e3"] @ self.components["e1"]
        self.components["sigma3"] = self.components["e1"] @ self.components["e2"]
        self.components["tau"] = (
            self.components["e1"] @ self.components["e2"] @ self.components["e3"]
        )
        self.components["id"] = self.components["e1"] @ self.components["e1"]

    def get_inverse(self, component):
        if component not in self.components:
            raise ValueError(f"Unknown component: {component}")
        return (
            -self.components[component]
            if "sigma" in component or component == "tau"
            else self.components[component]
        )

    # def extract_component(self, geom_number, component):
    #     return 1 / 4 * np.trace(geom_number @ self.get_inverse(component))

    def extract_component(self, geom_numbers, component):
        """
        Extracts the specified component from a batch of geometric numbers.

        Args:
            geom_numbers (np.array): Shape (..., 4, 4) representing geometric numbers.
            component (str): The name of the component to extract.

        Returns:
            np.array: Shape (...) with the extracted component values.
        """
        if component not in self.components:
            raise ValueError(f"Unknown component: {component}")

        inverse_component = self.get_inverse(component)  # Shape: (4, 4)
        # Perform batch matrix multiplication and trace computation
        # np.einsum computes traces over the last two axes of each batch element
        traces = np.einsum("...ij,ji->...", geom_numbers, inverse_component)
        return traces / 4

    def vector_to_geometric(self, coords_3d):
        return (
            coords_3d[0] * self.components["sigma1"]
            + coords_3d[1] * self.components["sigma2"]
            + coords_3d[2] * self.components["sigma3"]
        )

    def geometric_to_vector(self, geom_number):
        return np.array(
            [
                self.extract_component(geom_number, "sigma1"),
                self.extract_component(geom_number, "sigma2"),
                self.extract_component(geom_number, "sigma3"),
            ]
        )
