import numpy as np

from scipy.special import comb
from itertools import combinations
from sklearn.cluster import KMeans


''' Creating Weights using Lattice Resolution'''

class Referencepoints:
    """Class object for Predetermined reference points."""

    def __init__(self, lattice_resolution: int=4, number_of_objectives: int=None):
        """Create a Reference points object.
        

        A simplex lattice is formed

        Parameters
        ----------
        lattice_resolution : int
            Number of divisions along an axis when creating the simplex lattice.
        number_of_objectives : int
            Number of objectives.
        """
        number_of_points = comb(
            lattice_resolution + number_of_objectives - 1,
            number_of_objectives - 1,
            exact=True,
        )
        temp1 = range(1, number_of_objectives + lattice_resolution)
        temp1 = np.array(list(combinations(temp1, number_of_objectives - 1)))
        temp2 = np.array([range(number_of_objectives - 1)] * number_of_points)
        temp = temp1 - temp2 - 1
        weight = np.zeros((number_of_points, number_of_objectives), dtype=int)
        weight[:, 0] = temp[:, 0]
        for i in range(1, number_of_objectives - 1):
            weight[:, i] = temp[:, i] - temp[:, i - 1]
        weight[:, -1] = lattice_resolution - temp[:, -1]
        self.values = weight / lattice_resolution
        self.number_of_objectives = number_of_objectives
        self.lattice_resolution = lattice_resolution
        self.number_of_points = number_of_points
        self.normalize()
        self.initial_values = np.copy(self.values)
        # self.neighbouring_angles()
        # self.iteractive_adapt_1() Can use this for a priori preferences!

    def normalize(self):
        """Normalize the reference points to a unit hypersphere."""
        self.number_of_points = self.values.shape[0]
        norm = np.linalg.norm(self.values, axis=1)
        norm = np.repeat(norm, self.number_of_objectives).reshape(
            self.number_of_points, self.number_of_objectives
        )
        self.values = np.divide(self.values, norm)



