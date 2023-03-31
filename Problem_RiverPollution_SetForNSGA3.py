import numpy as np
from pymoo.core.problem import Problem

class RiverPollution_SetForNSGA3(Problem):

    def __init__(self, n_var=2, **kwargs):
        super().__init__(n_var=n_var, n_obj=4, n_ieq_constr=0, xl=0.3, xu=1, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        #f1 = x[:, 0]
        f1 = -1*(4.07*x[:, 0] + 2.27 * x[:, 1])
        f2 = -1*(2.6 + 0.03*x[:, 0] + 0.02*x[:, 1] + 0.01/(1.39 - x[:, 0]**2) + 0.3/(1.39 - x[:, 1] ** 2))
        f3 = -1 * (8.21 - 0.71 / (1.09 - x[:, 0]**2))
        f4 = (- 0.96 + (0.96 / (1.09 - x[:, 1]** 2)))

        out["F"] = np.column_stack([f1, f2, f3, f4])