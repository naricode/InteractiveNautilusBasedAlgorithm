from river_pollution import RiverPollution
from Population import Population
from sklearn.cluster import KMeans
from weights import Referencepoints
# Problem =
class ASF:
    
    """Evaluate ASF function of individual.

    Returns ASF function of individuals.

    Parameters
    ----------
    problem: our MOP
    objectives: objectives value
    weights: the representative weights
    reference_p: the reference point
    """

    def __init__(self, problem, objectives, weight, reference_p):
        self.problem=problem
        self.objectives=objectives
        self.weight=weight
        self.reference_p=reference_p

    def asf_fun(self):

        rho=0.1
        first_term = max([self.weight[i] * (self.objectives[i] - self.reference_p[i])
            for i in range(self.problem.num_of_objectives)])
                
        second_term = sum([self.weight[i]*(self.objectives[i] - self.reference_p[i])
            for i in range(self.problem.num_of_objectives)]) * rho
        
        return first_term + second_term  

